#!/usr/bin/env python3
"""Drive Raijin on the Arty A7 over UART and check it live against the golden model.

    pip install pyserial numpy
    python fpga/host/raijin_host.py --port COM5            (Windows)
    python fpga/host/raijin_host.py --port /dev/ttyUSB1    (Linux)

For each random case it loads Q, K, V, runs every dataflow and exponent mode,
reads the output and the counters, and compares with model/raijin_model.py.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "model"))
import raijin_model as rm  # noqa: E402

REG_CTRL, REG_OFFSET, REG_STATUS = 0x5000, 0x5001, 0x5002
REG_CYCLES, REG_SKIPS, REG_N, REG_D = 0x5004, 0x5008, 0x500C, 0x500D


class Raijin:
    def __init__(self, port, baud=115200):
        import serial  # pyserial
        self.s = serial.Serial(port, baud, timeout=2)
        self.s.reset_input_buffer()

    def _read_exact(self, n):
        data = self.s.read(n)
        if len(data) != n:
            raise TimeoutError(f"expected {n} bytes, got {len(data)}")
        return data

    def ping(self):
        self.s.write(b"P")
        return self._read_exact(1) == b"K"

    def write(self, addr, val):
        self.s.write(bytes([ord("W"), addr >> 8, addr & 0xFF, val & 0xFF]))

    def read(self, addr):
        self.s.write(bytes([ord("R"), addr >> 8, addr & 0xFF]))
        return self._read_exact(1)[0]

    def burst_write(self, addr, data):
        for i in range(0, len(data), 256):
            chunk = bytes(int(x) & 0xFF for x in data[i:i + 256])
            a = addr + i
            self.s.write(bytes([ord("B"), a >> 8, a & 0xFF, len(chunk) - 1]) + chunk)

    def burst_read(self, addr, n):
        out = b""
        for i in range(0, n, 256):
            m = min(256, n - i)
            a = addr + i
            self.s.write(bytes([ord("X"), a >> 8, a & 0xFF, m - 1]))
            out += self._read_exact(m)
        return out

    def read32(self, addr):
        b = self.burst_read(addr, 4)
        return int.from_bytes(b, "little")

    def load(self, q, eq, k, ek, v, ev):
        n, d = q.shape
        self.burst_write(0x0000, q.reshape(-1))
        self.burst_write(0x1000, k.reshape(-1))
        self.burst_write(0x2000, v.reshape(-1))
        self.burst_write(0x4000, eq)
        self.burst_write(0x4100, ek)
        self.burst_write(0x4200, ev)

    def run(self, n, d, fused, mode, offset=0):
        self.write(REG_OFFSET, offset)
        self.write(REG_CTRL, 1 | (int(fused) << 1) | (mode << 2))
        t0 = time.time()
        while not self.read(REG_STATUS) & 2:
            if time.time() - t0 > 2:
                raise TimeoutError("pass did not finish")
        o = np.frombuffer(self.burst_read(0x3000, n * d), dtype=np.int8).astype(np.int64).reshape(n, d)
        eo = np.frombuffer(self.burst_read(0x4300, n), dtype=np.uint8).astype(np.int64)
        return o, eo, self.read32(REG_CYCLES), self.read32(REG_SKIPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hw = Raijin(args.port, args.baud)
    assert hw.ping(), "no reply to ping; check the port and that the bitstream is loaded"
    n, d = hw.read(REG_N), hw.read(REG_D)
    print(f"Raijin found: N={n} D={d}")
    rng = np.random.default_rng(args.seed)
    names = {0: "exact", 1: "MF1", 2: "MF2"}
    fails = 0
    for case in range(args.cases):
        qf, kf, vf = rm.random_case(rng, n=n, d=d, scale=1.0 + case, outliers=bool(case % 2))
        tensors = rm.prepare(qf, kf, vf)
        hw.load(*tensors)
        ref = rm.reference(qf, kf, vf)
        for fused in (False, True):
            for mode in (0, 1, 2):
                exp = rm.attention(*tensors, fused=fused, mode=mode)
                o, eo, cycles, skips = hw.run(n, d, fused, mode)
                ok = np.array_equal(o, exp.o) and np.array_equal(eo, exp.eo) and skips == exp.skips
                got = rm.mx_dequantize(o, eo)
                cos = float(np.sum(got * ref) / np.linalg.norm(got) / np.linalg.norm(ref))
                fails += not ok
                print(f"case {case} {'fused  ' if fused else 'unfused'} {names[mode]:5s} "
                      f"cycles={cycles:4d} skips={skips:3d} cos={cos:.5f} "
                      f"{'MATCH' if ok else 'MISMATCH'}")
    print("all outputs match the golden model" if not fails else f"{fails} mismatches")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
