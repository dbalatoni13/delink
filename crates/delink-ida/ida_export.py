"""
delink IDA exporter
===================

Run this inside IDA (9.x) to export the *information* delink needs to split the
analysed binary into relocatable objects -- but **not** the bytes. The export
is compact and human-readable (one record per line) and contains the
architecture/segment layout, every function (boundaries + flags), every named
address, and the relocations IDA knows about. Data targets that IDA renders
with an auto-name but does not store in its named-address table are exported
with deterministic scalar names as well.

The bytes come from the original input binary, passed to delink on the command
line:

    delink ida-split <out.json> <original-binary> -o <dir>

Relocations are gathered from two IDA sources, deduplicated:
  * the fixup table (`ida_fixup`), and
  * address-bearing code/data operands, including offset-typed operands and
    32-bit absolute memory references backed by IDA data xrefs -- the only
    relocation record for images with no relocation table, e.g. fixed-base
    EXEs and original Xbox XBE files.
delink additionally reads the PE `.reloc` table from the binary (present in
DLLs; XBE inputs have no such table) and recovers rel32 calls/jumps with
iced-x86, resolving every target address through the exported name map.

Usage
-----
Headless (recommended)::

    A:\\IDA9.2\\idat.exe -A -S"crates\\delink-ida\\ida_export.py <out.json>" <database.i64>

  Optionally pass a third arg to also write a collapsed config grouping:
  `... ida_export.py <out.json> <config.json> [objname]`; pass `-` as the model
  path to write only the config.

Interactive: File -> Script file... and pick this script; it prompts for the
output path.

The JSON schema is documented in `crates/delink-ida/src/lib.rs`.
"""

import json
import os

import ida_auto
import ida_bytes
import ida_fixup
import ida_funcs
import ida_ida
import ida_kernwin
import ida_name
import ida_nalt
import ida_segment
import ida_ua
import ida_xref
import idaapi
import idautils
import idc

SCHEMA_VERSION = 3

BADADDR = idaapi.BADADDR


def _hex_address(value):
    return None if value is None else "0x%X" % int(value)


def _compact_json_dumps(value):
    """Pretty-print containers while keeping scalar records on one line."""
    def inline(value):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))

    def is_scalar(value):
        return not isinstance(value, (dict, list))

    def is_inline_object(value):
        if not isinstance(value, dict):
            return False
        if all(is_scalar(item) for item in value.values()):
            return True
        if all(
            is_scalar(item)
            or (isinstance(item, list) and all(is_scalar(child) for child in item))
            for item in value.values()
        ):
            return len(inline(value)) <= 180
        return False

    def render(value, depth):
        pad = "  " * depth
        child_pad = "  " * (depth + 1)
        if is_scalar(value):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, dict):
            if is_inline_object(value):
                return inline(value)
            if not value:
                return "{}"
            rows = [
                child_pad
                + json.dumps(key, ensure_ascii=False)
                + ": "
                + render(item, depth + 1)
                for key, item in value.items()
            ]
            return "{\n" + ",\n".join(rows) + "\n" + pad + "}"
        if not value:
            return "[]"
        if all(is_scalar(item) for item in value):
            tokens = [json.dumps(item, ensure_ascii=False) for item in value]
            if len(inline(value)) + len(pad) <= 110:
                return "[" + ", ".join(tokens) + "]"
            rows = []
            row = []
            row_len = 0
            for token in tokens:
                extra = len(token) + (2 if row else 0)
                if row and len(child_pad) + row_len + extra > 110:
                    rows.append(child_pad + ", ".join(row))
                    row = []
                    row_len = 0
                    extra = len(token)
                row.append(token)
                row_len += extra
            if row:
                rows.append(child_pad + ", ".join(row))
            return "[\n" + ",\n".join(rows) + "\n" + pad + "]"
        if all(
            isinstance(item, list) and all(is_scalar(child) for child in item)
            for item in value
        ):
            if len(inline(value)) + len(pad) <= 110:
                return inline(value)
        rows = [child_pad + render(item, depth + 1) for item in value]
        return "[\n" + ",\n".join(rows) + "\n" + pad + "]"

    return render(value, 0)


def _hexify_model_addresses(model):
    for key in ("image_base", "min_ea", "max_ea"):
        model["meta"][key] = _hex_address(model["meta"][key])
    for segment in model["segments"]:
        for key in ("start", "end"):
            segment[key] = _hex_address(segment[key])
    for function in model["functions"]:
        for key in ("start", "end", "thunk_target"):
            if key in function:
                function[key] = _hex_address(function[key])
    for name in model["names"]:
        name["addr"] = _hex_address(name["addr"])
    for relocation in model["relocations"]:
        for key in ("addr", "target"):
            relocation[key] = _hex_address(relocation[key])
    for table in model["jump_tables"]:
        for key in ("owner", "dispatch", "dispatch_addr", "start"):
            table[key] = _hex_address(table[key])
        for entry in table["entries"]:
            for key in ("addr", "target"):
                entry[key] = _hex_address(entry[key])
    return model


# ---------------------------------------------------------------------------
# small compatibility helpers (the inf_* getters moved around between versions)
# ---------------------------------------------------------------------------
def _call_first(*names, default=None):
    """Return the result of the first callable that exists, else `default`."""
    for name in names:
        fn = getattr(ida_ida, name, None)
        if fn is None:
            fn = getattr(idaapi, name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return default


def _procname():
    name = _call_first("inf_get_procname", "get_procName", default="")
    if not name:
        try:
            name = idaapi.get_inf_structure().procname  # very old fallback
        except Exception:
            name = ""
    return name or ""


def _app_bits():
    bits = _call_first("inf_get_app_bitness", default=None)
    if bits:
        return int(bits)
    if _call_first("inf_is_64bit", default=False):
        return 64
    if _call_first("inf_is_32bit_exactly", "inf_is_32bit", default=False):
        return 32
    if _call_first("inf_is_16bit", default=False):
        return 16
    return 32


def _is_big_endian():
    return bool(_call_first("inf_is_be", default=False))


def _image_base():
    try:
        return int(idaapi.get_imagebase())
    except Exception:
        return 0


def _min_ea():
    return int(_call_first("inf_get_min_ea", default=0) or 0)


def _max_ea():
    return int(_call_first("inf_get_max_ea", default=0) or 0)


def _arch(procname, bits):
    p = (procname or "").lower()
    if p in ("metapc", "8086", "80386p", "80386r", "80486p", "80486r", "80586p"):
        return "x86_64" if bits == 64 else "x86"
    if p.startswith("arm"):
        return "arm64" if bits == 64 else "arm"
    if p.startswith("ppc"):
        return "ppc64" if bits == 64 else "ppc"
    if p.startswith("mips"):
        return "mips64" if bits == 64 else "mips"
    return "unknown"


_FILETYPES = {
    getattr(ida_ida, "f_PE", -1): "PE",
    getattr(ida_ida, "f_ELF", -2): "ELF",
    getattr(ida_ida, "f_MACHO", -3): "MACHO",
    getattr(ida_ida, "f_COFF", -4): "COFF",
    getattr(ida_ida, "f_BIN", -5): "BIN",
}


def _filetype():
    ft = _call_first("inf_get_filetype", default=None)
    if ft is None:
        return "unknown"
    return _FILETYPES.get(int(ft), "other(%d)" % int(ft))


# ---------------------------------------------------------------------------
# segment class
# ---------------------------------------------------------------------------
def _seg_class(seg):
    t = seg.type
    if t == ida_segment.SEG_CODE:
        return "CODE"
    if t == ida_segment.SEG_BSS:
        return "BSS"
    if t == ida_segment.SEG_XTRN:
        return "XTRN"
    if t == ida_segment.SEG_DATA:
        # distinguish read-only (const) from writable data by permission bits
        if not (seg.perm & ida_segment.SEGPERM_WRITE):
            return "CONST"
        return "DATA"
    # fall back to IDA's class string (e.g. "CONST", "BSS")
    cls = ida_segment.get_segm_class(seg) or ""
    return cls.upper() or "DATA"


def export_segments():
    # Segment metadata only — the bytes come from the original input binary on
    # the delink command line, not from this export.
    out = []
    for ea in idautils.Segments():
        seg = ida_segment.getseg(ea)
        if seg is None or seg.end_ea <= seg.start_ea:
            continue
        cls = _seg_class(seg)
        bitness = {0: 16, 1: 32, 2: 64}.get(seg.bitness, 32)
        out.append(
            {
                "name": ida_segment.get_segm_name(seg) or "",
                "start": int(seg.start_ea),
                "end": int(seg.end_ea),
                "perm_r": bool(seg.perm & ida_segment.SEGPERM_READ),
                "perm_w": bool(seg.perm & ida_segment.SEGPERM_WRITE),
                "perm_x": bool(seg.perm & ida_segment.SEGPERM_EXEC),
                "class": cls,
                "bitness": bitness,
            }
        )
    return out


# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------
def _thunk_target(func):
    try:
        ea = ida_funcs.calc_thunk_func_target(func, None)
        if isinstance(ea, tuple):  # some builds return (ea, ...)
            ea = ea[0]
        if ea is not None and ea != BADADDR:
            return int(ea)
    except Exception:
        pass
    return None


def export_functions():
    out = []
    for ea in idautils.Functions():
        func = ida_funcs.get_func(ea)
        if func is None:
            continue
        flags = func.flags
        is_thunk = bool(flags & ida_funcs.FUNC_THUNK)
        out.append(
            {
                "start": int(func.start_ea),
                "size": "0x%X" % (int(func.end_ea) - int(func.start_ea)),
                "name": ida_funcs.get_func_name(ea) or ("sub_%X" % ea),
                "thunk": is_thunk,
                "lib": bool(flags & ida_funcs.FUNC_LIB),
                "static": bool(flags & ida_funcs.FUNC_STATICDEF),
                "public": bool(ida_name.is_public_name(ea)),
                "thunk_target": _thunk_target(func) if is_thunk else None,
            }
        )
    return out


def _switch_dispatch_field(ea, jumps):
    """Return the encoded address field that names a switch table."""
    insn = ida_ua.insn_t()
    if ida_ua.decode_insn(insn, ea) <= 0:
        return None
    data_refs = {int(ref) for ref in idautils.DataRefsFrom(ea)}
    for op in insn.ops:
        if op.type == ida_ua.o_void:
            break
        if op.offb == 0:
            continue
        if int(op.addr) == jumps or jumps in data_refs:
            return int(ea + op.offb)
    return None


def export_jump_tables(_ptr_size):
    """Export IDA switch metadata independently of function boundaries.

    A compiler may place the table inside the range IDA assigned to a function,
    or immediately after its last instruction.  Recording the range explicitly
    lets delink avoid disassembling data and extend the emitted function when
    the latter layout is used.
    """
    out = []
    seen = set()
    for func_ea in idautils.Functions():
        func = ida_funcs.get_func(func_ea)
        if func is None:
            continue
        for ea in idautils.FuncItems(func.start_ea):
            si = ida_nalt.switch_info_t()
            try:
                # IDA 9.2's Python compatibility wrapper returns ``si`` (or
                # None) even when called with the legacy two-argument form;
                # older releases returned an integer status. Accept both.
                switch_result = ida_nalt.get_switch_info(si, ea)
                if switch_result is None or switch_result is False:
                    continue
                if isinstance(switch_result, int) and switch_result <= 0:
                    continue
            except Exception:
                continue
            jumps = int(si.jumps)
            count = int(si.get_jtable_size())
            elem_size = int(si.get_jtable_element_size())
            key = (int(func.start_ea), jumps)
            if jumps == BADADDR or count <= 0 or elem_size not in (4, 8) or key in seen:
                continue
            seen.add(key)

            calculated = []
            try:
                calculated = [int(x) for x in ida_xref.calc_switch_cases(ea, si).targets]
            except Exception:
                pass

            entries = []
            for index in range(count):
                entry_ea = jumps + index * elem_size
                refs = [int(x) for x in idautils.DataRefsFrom(entry_ea)]
                refs.extend(int(x) for x in idautils.CodeRefsFrom(entry_ea, False))
                target = refs[0] if refs else None
                if target is None and index < len(calculated):
                    target = calculated[index]
                if target is None:
                    # Absolute tables are the normal x86 form. Relative/custom
                    # tables should have xrefs or calculated switch targets.
                    target = int(
                        ida_bytes.get_qword(entry_ea)
                        if elem_size == 8
                        else ida_bytes.get_dword(entry_ea)
                    )
                entries.append({"addr": int(entry_ea), "target": target})

            out.append(
                {
                    "owner": int(func.start_ea),
                    "dispatch": int(ea),
                    "dispatch_addr": _switch_dispatch_field(ea, jumps),
                    "start": jumps,
                    "entry_size": elem_size,
                    "entries": entries,
                    "name": ida_name.get_name(jumps) or ("jpt_%X" % jumps),
                }
            )
    out.sort(key=lambda item: (item["owner"], item["start"]))
    return out


# ---------------------------------------------------------------------------
# names (the full address -> symbol map used to resolve relocation targets)
# ---------------------------------------------------------------------------
def _default_data_name(address, size):
    """Match IDA's conventional name for an unnamed scalar at `address`."""
    prefix = {1: "byte", 2: "word", 4: "dword", 8: "qword"}.get(size, "data")
    return "%s_%X" % (prefix, address)


def export_names(relocations=None):
    """Export IDA names and synthesize names for unnamed data relocations.

    Hex-Rays can render an auto-name such as ``dword_893D58`` for a data
    reference even when the address is not present in IDA's named-address
    table. Keep those relocation targets editable in the exported JSON by
    adding a deterministic scalar name when no real IDA name exists.
    """
    out = []
    known = set()
    for ea, name in idautils.Names():
        if not name:
            continue
        ea = int(ea)
        known.add(ea)
        func = ida_funcs.get_func(ea)
        seg = ida_segment.getseg(ea)
        out.append(
            {
                "addr": ea,
                "name": name,
                "public": bool(ida_name.is_public_name(ea)),
                "weak": bool(ida_name.is_weak_name(ea)),
                "is_func": bool(func is not None and func.start_ea == ea),
                **({"size": "0x%X" % max(1, int(ida_bytes.get_item_size(ea)))}
                   if func is None and seg is not None and _seg_class(seg) in ("DATA", "CONST", "BSS") else {}),
            }
        )

    # IDA's decompiler may display an auto-generated dword_/qword_ label for a
    # data item that has no entry in idautils.Names(). Relocations still carry
    # the exact target, so use them to make such targets first-class symbols.
    targets = {}
    for relocation in relocations or ():
        target = int(relocation.get("target") or 0)
        if target in known or target == 0 or target == BADADDR:
            continue
        seg = ida_segment.getseg(target)
        if seg is None or _seg_class(seg) not in ("CONST", "DATA", "BSS"):
            continue
        size = int(relocation.get("size") or 0)
        targets[target] = max(size, targets.get(target, 0))

    for target, size in sorted(targets.items()):
        # Prefer a name that IDA can resolve directly, even if the iterator
        # omitted it for this kind of unnamed data item.
        try:
            name = ida_name.get_name(target) or ""
        except Exception:
            name = ""
        out.append(
            {
                "addr": target,
                "name": name or _default_data_name(target, size),
                "public": False,
                "weak": False,
                "is_func": False,
                "size": "0x%X" % max(1, size),
            }
        )

    out.sort(key=lambda item: item["addr"])
    # IDA can place a label inside an array or struct. Keep exported sizes
    # disjoint while leaving the user free to expand a symbol after removing
    # interior labels from the JSON.
    for index, item in enumerate(out):
        if "size" not in item:
            continue
        seg = ida_segment.getseg(item["addr"])
        limit = int(seg.end_ea) if seg is not None else item["addr"] + 1
        if index + 1 < len(out):
            limit = min(limit, out[index + 1]["addr"])
        item["size"] = "0x%X" % min(int(item["size"], 16), max(1, limit - item["addr"]))
    return out


# ---------------------------------------------------------------------------
# relocations (IDA's fixup table -- absolute address fixups the loader applied)
# ---------------------------------------------------------------------------
_FIXUP_WIDTH = {
    getattr(ida_fixup, "FIXUP_OFF8", -1): 1,
    getattr(ida_fixup, "FIXUP_OFF16", -2): 2,
    getattr(ida_fixup, "FIXUP_OFF32", -3): 4,
    getattr(ida_fixup, "FIXUP_OFF64", -4): 8,
}


def _fixup_type_name(ftype):
    base = ftype & getattr(ida_fixup, "FIXUP_MASK", 0xF)
    for attr in ("FIXUP_OFF8", "FIXUP_OFF16", "FIXUP_OFF32", "FIXUP_OFF64"):
        if base == getattr(ida_fixup, attr, None):
            return attr[len("FIXUP_"):]
    return "T%d" % base


def export_relocations():
    out = []
    ea = ida_fixup.get_first_fixup_ea()
    while ea != BADADDR:
        fd = ida_fixup.fixup_data_t()
        ok = False
        try:
            ok = ida_fixup.get_fixup(fd, ea)
        except TypeError:
            # older signature: get_fixup(ea, fd)
            try:
                ok = ida_fixup.get_fixup(ea, fd)
            except Exception:
                ok = False
        if ok:
            ftype = fd.get_type()
            base = ftype & getattr(ida_fixup, "FIXUP_MASK", 0xF)
            width = _FIXUP_WIDTH.get(base, 0)
            # Target VA: prefer the fixup descriptor, fall back to reading the
            # already-relocated value out of the loaded image.
            target = None
            try:
                target = int(fd.get_base()) + int(fd.off)
            except Exception:
                target = None
            if target is None or target == 0:
                if width == 8:
                    target = int(ida_bytes.get_qword(ea))
                elif width == 4:
                    target = int(ida_bytes.get_dword(ea))
                elif width == 2:
                    target = int(ida_bytes.get_word(ea))
            out.append(
                {
                    "addr": int(ea),
                    "type": _fixup_type_name(ftype),
                    "size": width,
                    "target": target if target is not None else 0,
                }
            )
        ea = ida_fixup.get_next_fixup_ea(ea)
    return out


def _encoded_field_size(insn, op):
    """Width of this operand's encoded field, not its value's data type.

    For example, `mov dword ptr [eax+3Ch], offset name` has a one-byte
    displacement followed by a four-byte immediate. Both operands have a
    dword *data type*, but only the immediate can hold an OFF32 relocation.
    """
    next_field = min(
        (other.offb for other in insn.ops if op.offb < other.offb < insn.size),
        default=insn.size,
    )
    return next_field - op.offb


def export_offset_relocations(ptr_size):
    """Absolute relocations derived from IDA's address-bearing operands.

    IDA does not mark every absolute memory operand with `is_off()`. In
    particular, indexed PE32 operands such as `global[index*4]` can have a data
    xref and a rendered name while remaining an ordinary o_displ/o_mem operand.
    Recover those from the decoded operand plus IDA's outgoing data xrefs.

    For each address-bearing operand, record the exact encoded field address
    (`insn ea + op.offb` for code, the item ea for data), its width, and target
    VA. The 32-bit memory-xref path is deliberately limited to PE32-style
    absolute displacements; x86-64 RIP-relative fields are not absolute.
    """
    out = []
    for seg_ea in idautils.Segments():
        seg = ida_segment.getseg(seg_ea)
        if seg is None:
            continue
        ea = seg.start_ea
        end = seg.end_ea
        while ea < end and ea != BADADDR:
            f = ida_bytes.get_full_flags(ea)
            offsets = (ida_bytes.is_off(f, 0), ida_bytes.is_off(f, 1))
            if ida_bytes.is_code(f):
                insn = ida_ua.insn_t()
                if ida_ua.decode_insn(insn, ea) > 0:
                    data_refs = {int(ref) for ref in idautils.DataRefsFrom(ea)}
                    for n in (0, 1):
                        op = insn.ops[n]
                        if op.type == ida_ua.o_void or op.offb == 0:
                            continue  # no locatable encoded field

                        is_offset = offsets[n]
                        field = ea + op.offb
                        field_size = _encoded_field_size(insn, op)
                        is_abs_memory = (
                            ptr_size == 4
                            and op.type in (ida_ua.o_mem, ida_ua.o_displ)
                            and bool(data_refs)
                            and field_size == 4
                        )
                        if (not is_offset and not is_abs_memory) or field_size not in (4, 8):
                            continue

                        if is_abs_memory and not is_offset:
                            # An instruction's data xrefs also include other
                            # operands. Require the encoded displacement (or
                            # IDA's decoded address) to name this xref.
                            operand_target = int(op.addr)
                            encoded_target = int(ida_bytes.get_dword(field))
                            if operand_target in data_refs:
                                target = operand_target
                            elif encoded_target in data_refs:
                                target = encoded_target
                            else:
                                continue
                            size = 4
                        else:
                            size = field_size
                            if size == 8:
                                target = int(ida_bytes.get_qword(field))
                            else:
                                target = int(ida_bytes.get_dword(field))

                        out.append(
                            {
                                "addr": int(field),
                                "type": "OFF%d" % (size * 8),
                                "size": size,
                                "target": target,
                            }
                        )
            else:
                data_refs = {int(ref) for ref in idautils.DataRefsFrom(ea)}
                if offsets[0]:  # data offset item — the field is the item itself
                    size = ptr_size
                    if size == 8:
                        target = int(ida_bytes.get_qword(ea))
                    else:
                        size = 4
                        target = int(ida_bytes.get_dword(ea))
                    out.append(
                        {
                            "addr": int(ea),
                            "type": "OFF%d" % (size * 8),
                            "size": size,
                            "target": target,
                        }
                    )
                elif data_refs:
                    # IDA can attach a data xref inside a structure/array to
                    # the item's head instead of the pointer field. Scan the
                    # item's bytes for pointer-width values matching those
                    # xref targets, then export the relocation at the field.
                    item_size = int(ida_bytes.get_item_size(ea))
                    if ptr_size in (4, 8) and item_size >= ptr_size:
                        for offset in range(item_size - ptr_size + 1):
                            field = ea + offset
                            if ptr_size == 8:
                                target = int(ida_bytes.get_qword(field))
                            else:
                                target = int(ida_bytes.get_dword(field))
                            if target not in data_refs:
                                continue
                            out.append(
                                {
                                    "addr": int(field),
                                    "type": "OFF%d" % (ptr_size * 8),
                                    "size": ptr_size,
                                    "target": target,
                                }
                            )
            nh = ida_bytes.next_head(ea, end)
            if nh <= ea:
                break
            ea = nh
    return out


def build_relocations(ptr_size):
    """Combine IDA fixups and inferred address operands (dedup by address)."""
    by_addr = {}
    for r in export_relocations():
        by_addr[r["addr"]] = r
    for r in export_offset_relocations(ptr_size):
        by_addr.setdefault(r["addr"], r)
    return [by_addr[a] for a in sorted(by_addr)]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_model():
    procname = _procname()
    bits = _app_bits()
    ptr_size = 8 if bits == 64 else 4
    jump_tables = export_jump_tables(ptr_size)
    relocations = build_relocations(ptr_size)
    return _hexify_model_addresses({
        "delink_ida_version": SCHEMA_VERSION,
        "meta": {
            "arch": _arch(procname, bits),
            "procname": procname,
            "bits": bits,
            "endian": "big" if _is_big_endian() else "little",
            "image_base": _image_base(),
            "min_ea": _min_ea(),
            "max_ea": _max_ea(),
            "filetype": _filetype(),
            "input_file": ida_nalt.get_input_file_path() or "",
        },
        "segments": export_segments(),
        "functions": export_functions(),
        "jump_tables": jump_tables,
        "names": export_names(relocations),
        "relocations": relocations,
    })


def _default_object_name():
    """Default output-object name for the config: `<database basename>.obj`."""
    db = idc.get_idb_path() or ida_nalt.get_input_file_path() or "output"
    base = os.path.basename(db)
    root, ext = os.path.splitext(base)
    if ext.lower() in (".i64", ".idb"):
        base = root
    return base + ".obj"


def build_config(obj_name):
    """Collapsed grouping config: every function start address → `obj_name`.

    The empty range lists make the editable whole-function, `.rdata`, `.data`,
    and logical `.bss` range syntax visible in the generated config. Ranges are
    half-open `[start, end)` pairs; function ranges must contain complete
    functions. A `.bss` range may select a BSS segment or a zero-initialized
    tail that IDA reports inside DATA.
    """
    addrs = []
    for ea in idautils.Functions():
        func = ida_funcs.get_func(ea)
        if func is None or func.end_ea <= func.start_ea:
            continue
        if func.flags & ida_funcs.FUNC_TAIL:
            continue
        addrs.append(_hex_address(func.start_ea))
    return {
        obj_name: {
            "functions": addrs,
            "function_ranges": [],
            "rdata": [],
            "data": [],
            "bss": [],
        }
    }


def main():
    # Make sure auto-analysis is complete before we read the database.
    ida_auto.auto_wait()

    # Args (from `idat -S"ida_export.py <model> [config] [objname]"`):
    #   <model>   path for the full model JSON (segments+bytes+functions+...).
    #             Pass "-" (or "none") to skip it and only write the config.
    #   [config]  optional path for the grouping/config JSON (new format).
    #   [objname] optional object name for the config (default <db>.obj).
    argv = list(idc.ARGV) if idc.ARGV else []
    model_out = argv[1] if len(argv) > 1 else None
    config_out = argv[2] if len(argv) > 2 else None
    obj_name = argv[3] if len(argv) > 3 else None

    if not model_out and not config_out and ida_kernwin.is_idaq():
        default = (ida_nalt.get_input_file_path() or "delink") + ".delink.json"
        model_out = ida_kernwin.ask_file(True, default, "Export delink JSON")

    skip_model = (model_out is None) or (model_out in ("-", "none", ""))
    wrote = []

    if not skip_model:
        model = build_model()
        with open(model_out, "w", encoding="utf-8") as fh:
            fh.write(_compact_json_dumps(model) + "\n")
        wrote.append(
            "model %d functions/%d names/%d relocs/%d segments -> %s"
            % (
                len(model["functions"]),
                len(model["names"]),
                len(model["relocations"]),
                len(model["segments"]),
                model_out,
            )
        )

    if config_out:
        cfg = build_config(obj_name or _default_object_name())
        with open(config_out, "w", encoding="utf-8") as fh:
            fh.write(_compact_json_dumps(cfg) + "\n")
        nfunc = sum(len(v["functions"]) for v in cfg.values())
        wrote.append("config %d functions -> %s" % (nfunc, config_out))

    if wrote:
        for line in wrote:
            ida_kernwin.msg("delink: wrote " + line + "\n")
    else:
        ida_kernwin.msg("delink: no output path supplied; nothing written\n")

    # When running headless (idat -A -S...), close IDA so the process exits.
    if not ida_kernwin.is_idaq():
        idaapi.qexit(0)


if __name__ == "__main__":
    main()
