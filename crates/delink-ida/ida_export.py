"""
delink IDA exporter
===================

Run this inside IDA (9.x) to export the *information* delink needs to split the
analysed binary into relocatable objects -- but **not** the bytes.  The export
is small and human-readable (pretty-printed JSON) and contains: the
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
    EXEs.
delink additionally reads the PE `.reloc` table from the binary (present in the
DLLs) and recovers rel32 calls/jumps with iced-x86, resolving every target
address through the exported name map.

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
import idaapi
import idautils
import idc

SCHEMA_VERSION = 1

BADADDR = idaapi.BADADDR


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
                "end": int(func.end_ea),
                "name": ida_funcs.get_func_name(ea) or ("sub_%X" % ea),
                "thunk": is_thunk,
                "lib": bool(flags & ida_funcs.FUNC_LIB),
                "static": bool(flags & ida_funcs.FUNC_STATICDEF),
                "public": bool(ida_name.is_public_name(ea)),
                "thunk_target": _thunk_target(func) if is_thunk else None,
            }
        )
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
        out.append(
            {
                "addr": ea,
                "name": name,
                "public": bool(ida_name.is_public_name(ea)),
                "weak": bool(ida_name.is_weak_name(ea)),
                "is_func": bool(func is not None and func.start_ea == ea),
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
            }
        )

    out.sort(key=lambda item: item["addr"])
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


def _dtype_size(dtype, default):
    try:
        sz = ida_ua.get_dtype_size(dtype)
        if sz in (1, 2, 4, 8):
            return sz
    except Exception:
        pass
    return default


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
                        is_abs_memory = (
                            ptr_size == 4
                            and op.type in (ida_ua.o_mem, ida_ua.o_displ)
                            and bool(data_refs)
                        )
                        if not is_offset and not is_abs_memory:
                            continue

                        field = ea + op.offb
                        if is_abs_memory and not is_offset:
                            # op.addr is normally the base VA. Use the xref as a
                            # fallback for processor-module variants that store
                            # only the displacement in the operand structure.
                            operand_target = int(op.addr)
                            if operand_target in data_refs:
                                target = operand_target
                            elif len(data_refs) == 1:
                                target = next(iter(data_refs))
                            else:
                                continue  # ambiguous: do not guess a target
                            size = 4
                        else:
                            size = _dtype_size(op.dtype, ptr_size)
                            if size == 8:
                                target = int(ida_bytes.get_qword(field))
                            else:
                                size = 4
                                target = int(ida_bytes.get_dword(field))

                        out.append(
                            {
                                "addr": int(field),
                                "type": "OFF%d" % (size * 8),
                                "size": size,
                                "target": target,
                            }
                        )
            elif offsets[0]:  # data offset item — the field is the item itself
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
    relocations = build_relocations(8 if bits == 64 else 4)
    return {
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
        "names": export_names(relocations),
        "relocations": relocations,
    }


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
        addrs.append(int(func.start_ea))
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
            # Pretty-printed so the export is human-readable and editable.
            json.dump(model, fh, indent=2)
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
            json.dump(cfg, fh, indent=2)
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
