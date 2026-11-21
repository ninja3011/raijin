#!/usr/bin/env python3
"""Cut a VCD down to one interval where a 1-bit top-level signal is high.

  vcd_window.py in.vcd out.vcd [signal=busy] [interval_index=0]

The output starts at the rising edge with a full $dumpvars of every signal's
value at that time, and ends at the falling edge, so OpenSTA's read_vcd
averages activity over that window only.
"""
import sys


def header_and_ids(lines, signal):
    hdr, ident, depth = [], None, 0
    for ln in lines:
        hdr.append(ln)
        t = ln.split()
        if t[:1] == ["$scope"]:
            depth += 1
        elif t[:1] == ["$upscope"]:
            depth -= 1
        elif t[:1] == ["$var"] and depth == 1 and t[4] == signal and t[2] == "1":
            ident = t[3]
        elif t[:1] == ["$enddefinitions"]:
            return hdr, ident
    raise SystemExit("no $enddefinitions")


def changes(fh):
    """Yield (time, id, text) for each value change after the header."""
    now = 0
    for ln in fh:
        s = ln.strip()
        if not s or s.startswith("$dumpvars") or s == "$end" or s.startswith("$dump"):
            continue
        if s[0] == "#":
            now = int(s[1:])
            yield now, None, None
        elif s[0] in "01xzXZ":
            yield now, s[1:], s
        elif s[0] in "bBrR":
            v, i = s.split()
            yield now, i, s


def main(src, dst, signal="busy", index=0):
    index = int(index)
    with open(src) as fh:
        hdr, bid = header_and_ids(fh, signal)
        if bid is None:
            raise SystemExit(f"top-level 1-bit signal {signal!r} not found")
        start = end = None
        n = -1
        for now, i, s in changes(fh):
            if i == bid:
                if s[0] == "1":
                    n += 1
                    if n == index:
                        start = now
                elif s[0] == "0" and start is not None and end is None:
                    end = now
                    break
    if start is None or end is None:
        raise SystemExit("interval not found")
    print(f"{signal} interval {index}: {start} .. {end} ({end - start} time units)")
    vals = {}
    with open(src) as fh, open(dst, "w") as out:
        header_and_ids(fh, signal)
        out.writelines(hdr)
        emitted = False
        for now, i, s in changes(fh):
            if now >= end:
                out.write(f"#{end}\n")
                break
            if now < start:
                if i is not None:
                    vals[i] = s
                continue
            if not emitted:
                out.write(f"#{start}\n$dumpvars\n")
                out.writelines(v + "\n" for v in vals.values())
                out.write("$end\n")
                emitted = True
            if i is None:
                if now > start:
                    out.write(f"#{now}\n")
            else:
                out.write(s + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
