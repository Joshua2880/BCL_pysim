import random
from typing import NewType
from enum import Enum
from enum import IntEnum
import numpy as np


class ZeroByZeroDivisionError(ArithmeticError):
    pass


class RPU:
    bcl = NewType('bcl', int)

    bcl_width = 64
    bcl_msb = 1 << (bcl_width - 1)

    register_msb = 1 << (4 * bcl_width - 1)

    class Op(Enum):
        NOP = 0
        NEW = 1
        FRAC = 2
        LFT = 3
        BLFT = 4

    class Flag(IntEnum):
        INVALID = 1 << 0
        DIV_BY_ZERO = 1 << 1
        INEXACT = 1 << 2
        ERROR = 1 << 3

    def __init__(self):
        self.t_reg = np.array([[[0, 0],
                                [0, 0]],
                               [[0, 0],
                                [0, 0]]], dtype='object')

        self.in_cntr = 0
        self.out_cntr = 0

        self.emission_ready = False
        self.emission_val = False

        self.op = RPU.Op.NOP

        self.flags = 0

    def init_tensor(self, a: int, b: int, c: int, d: int,
                    e: int, f: int, g: int, h: int):
        self.t_reg = np.array([[[a, b],
                                [c, d]],
                               [[e, f],
                                [g, h]]], dtype='object')

    def init_cntrs(self):
        self.in_cntr = 0
        self.out_cntr = 0

    # combinational logic in hardware version
    def update_state(self):
        self.flags = 0

        x_interval = np.array([[1, 1], [0, 0]])
        y_interval = np.array([[1, 1], [0, 0]])

        if self.op == RPU.Op.LFT:
            x_interval = np.array([[1, 1], [0, 1]]) if self.in_cntr >= 2 else np.array([[1, 0], [0, 1]])
        elif self.op == RPU.Op.BLFT:
            x_interval = np.array([[1, 1], [0, 1]]) if self.in_cntr >= 2 else np.array([[1, 0], [0, 1]])
            y_interval = np.array([[1, 1], [0, 1]]) if self.in_cntr >= 2 else np.array([[1, 0], [0, 1]])

        out_interval = np.einsum("ijl, jk", self.t_reg, x_interval)
        out_interval = np.einsum("ijk, kl", out_interval, y_interval)

        out_interval *= 2 * (out_interval[1] >= 0) - 1

        zeros = np.array((2, 2), dtype=np.bool_)
        ones = np.array((2, 2), dtype=np.bool_)

        if self.out_cntr == 0:
            zeros = np.logical_and(out_interval[0] >= 0, np.logical_or(out_interval[0] != 0, out_interval[1] != 0))
            ones = out_interval[0] < 0
        elif self.out_cntr == 1:
            zeros = np.abs(out_interval[1]) <= np.abs(out_interval[0])
            ones = np.abs(out_interval[1]) >= np.abs(out_interval[0])
        else:
            zeros = np.abs(out_interval[0]) <= 2 * np.abs(out_interval[1])
            ones = np.abs(out_interval[0]) >= 2 * np.abs(out_interval[1])

        zero = np.all(zeros)
        one = np.all(ones)

        ingest_finished = self.op == RPU.Op.NEW or self.in_cntr == RPU.bcl_width

        self.emission_ready = (ingest_finished or zero or one) and self.out_cntr < RPU.bcl_width
        if not ingest_finished:
            self.emission_val = not zero
        else:
            if np.any(out_interval[:, 0, 0]):
                self.emission_val = not zeros[0, 0]
            else:
                if zeros[0, 1] and zeros[1, 0]:
                    self.emission_val = False
                elif ones[0, 1] and ones[1, 0]:
                    self.emission_val = True
                else:
                    if out_interval[0, 0, 0] == 0 and out_interval[1, 0, 0] == 0:
                        self.flags |= self.Flag.INVALID
                    raise ZeroByZeroDivisionError

    def new(self, n: int, d: int) -> bcl:
        if n == 0 and d == 0:
            raise ZeroByZeroDivisionError
        self.op = RPU.Op.NEW
        self.init_tensor(n, 0, 0, 0, d, 0, 0, 0)
        self.init_cntrs()

        z = RPU.bcl(0)

        self.update_state()
        while self.emission_ready:
            z = self.emmit_z(z)
            self.out_cntr += 1
            self.normalize()
            self.update_state()

        return z

    def frac(self, x: bcl) -> (int, int):
        self.op = RPU.Op.FRAC
        self.init_tensor(1, 0, 0, 0, 0, 0, 1, 0)
        self.init_cntrs()

        while self.in_cntr < RPU.bcl_width:
            x = self.ingest_x(x)
            self.in_cntr += 1
            self.normalize()
        return self.t_reg[0, 0, 0], self.t_reg[1, 0, 0]

    def lft(self, x: bcl, a: int, b: int, c: int, d: int) -> bcl:
        self.op = RPU.Op.LFT
        self.init_tensor(a, 0, b, 0, c, 0, d, 0)
        self.init_cntrs()

        z = RPU.bcl(0)

        while self.out_cntr < RPU.bcl_width:
            x = self.ingest_x(x)
            self.in_cntr += 1
            self.normalize()

            self.update_state()
            while self.emission_ready:
                z = self.emmit_z(z)
                self.out_cntr += 1
                self.normalize()
                self.update_state()

        return z

    def blft(self, x: bcl, y: bcl, a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int) -> bcl:
        self.op = RPU.Op.BLFT
        self.init_tensor(a, b, c, d, e, f, g, h)
        self.init_cntrs()

        z = RPU.bcl(0)

        while self.out_cntr < RPU.bcl_width:
            x = self.ingest_x(x)
            y = self.ingest_y(y)
            self.in_cntr += 1
            self.normalize()

            self.update_state()
            while self.emission_ready:
                z = self.emmit_z(z)
                self.out_cntr += 1
                self.normalize()
                self.update_state()

        return z

    def ingest_x(self, x: bcl) -> bcl:
        ingestion_val = x & RPU.bcl_msb

        op_tensor = np.array([[1, 0], [0, 1]])
        if self.in_cntr == 0:
            if ingestion_val:
                op_tensor = np.array([[-1, 0], [0, 1]])
        elif self.in_cntr == 1:
            if ingestion_val:
                op_tensor = np.array([[0, 1], [1, 0]])
        elif self.in_cntr < RPU.bcl_width:
            if ingestion_val:
                op_tensor = np.array([[2, 0], [0, 1]])
            else:
                op_tensor = np.array([[1, 1], [1, 0]])

        self.t_reg = np.einsum("ijl, jk", self.t_reg, op_tensor)

        return RPU.bcl(x << 1)

    def ingest_y(self, y: bcl) -> bcl:
        ingestion_val = y & RPU.bcl_msb

        op_tensor = np.array([[1, 0], [0, 1]])
        if self.in_cntr == 0:
            if ingestion_val:
                op_tensor = np.array([[-1, 0], [0, 1]])
        elif self.in_cntr == 1:
            if ingestion_val:
                op_tensor = np.array([[0, 1], [1, 0]])
        elif self.in_cntr < RPU.bcl_width:
            if ingestion_val:
                op_tensor = np.array([[2, 0], [0, 1]])
            else:
                op_tensor = np.array([[1, 1], [1, 0]])

        self.t_reg = np.einsum("ijk, kl", self.t_reg, op_tensor)

        return RPU.bcl(y << 1)

    def emmit_z(self, z: bcl) -> bcl:
        op_tensor = np.array([[1, 0], [0, 1]])
        if self.out_cntr == 0:
            if self.emission_val:
                op_tensor = np.array([[-1, 0], [0, 1]])
        elif self.out_cntr == 1:
            if self.emission_val:
                op_tensor = np.array([[0, 1], [1, 0]])
        else:
            if self.emission_val:
                op_tensor = np.array([[1, 0], [0, 2]])
            else:
                op_tensor = np.array([[0, 1], [1, -1]])

        self.t_reg = np.einsum("ij, jkl", op_tensor, self.t_reg)

        return RPU.bcl((z << 1) | self.emission_val)

    def normalize(self):
        bits_set = np.bitwise_or.reduce(self.t_reg, axis=(0, 1, 2))
        self.t_reg >>= (bits_set & -bits_set).bit_length()-1

    def add(self, x: bcl, y: bcl) -> bcl:
        return self.blft(x, y, 0, 1, 1, 0, 0, 0, 0, 1)

    def sub(self, x: bcl, y: bcl) -> bcl:
        return self.blft(x, y, 0, 1, -1, 0, 0, 0, 0, 1)

    def mul(self, x: bcl, y: bcl) -> bcl:
        return self.blft(x, y, 1, 0, 0, 0, 0, 0, 0, 1)

    def div(self, x: bcl, y: bcl) -> bcl:
        return self.blft(x, y, 0, 1, 0, 0, 0, 0, 1, 0)

    @staticmethod
    def bcl2int(x: bcl) -> int:
        width_mask = ~(~0 << RPU.bcl_width)
        grey_code = x ^ (width_mask >> 1)
        result = 0
        for i in range(RPU.bcl_width):
            result ^= grey_code >> i
        return (result >> 1) + (result & 1)

    @staticmethod
    def int2bcl(x: int) -> bcl:
        width_mask = ~(~0 << RPU.bcl_width)
        return (x << 1) ^ x ^ (width_mask >> 1)

    @staticmethod
    def quickcmp(x: bcl, y: bcl) -> bool:
        mask = ((~x & (x + 1)) << 1) & ~(~0 << RPU.bcl_width)
        return ~(~(x ^ y) | mask) == 0 or mask == 0

    @staticmethod
    def abs(x: bcl) -> bcl:
        return RPU.bcl(~RPU.bcl_msb & x)

    @staticmethod
    def gtz(x: bcl) -> bcl:
        return x != ~RPU.bcl_msb and not x & RPU.bcl_msb


if __name__ == "__main__":
    bit_f = "{0:0%db}" % RPU.bcl_width
    rpu = RPU()

    x = rpu.new(-107, 85)
    y = rpu.new(-2844456, 110075)
    result = rpu.div(x, y)
    assert (RPU.quickcmp(result, 0x7D21B9CFB07FFFFF))

    x = rpu.new(77617, 1)
    y = rpu.new(33096, 1)

    x_2 = rpu.mul(x, x)
    y_2 = rpu.mul(y, y)
    y_4 = rpu.mul(y_2, y_2)
    y_6 = rpu.mul(y_4, y_2)

    n, d = rpu.frac(x_2)
    print(f"{n} / {d}")

    c1 = rpu.new(1335, 4)
    c2 = rpu.new(11, 1)
    c4 = rpu.new(121, 1)
    c5 = rpu.new(2, 1)
    c6 = rpu.new(11, 2)
    c7 = rpu.new(1, 2)

    e1 = rpu.mul(c1, y_6)

    e2_1 = rpu.mul(c2, x_2)
    e2_1 = rpu.mul(e2_1, y_2)

    e2_3 = rpu.mul(c4, y_4)

    try:
        e2 = rpu.sub(e2_1, y_6)
        assert (False)
        e2 = rpu.sub(e2, e2_3)
        e2 = rpu.sub(e2, c5)
        e2 = rpu.mul(e2, x_2)

        e3 = rpu.mul(c6, y)

        e4 = rpu.mul(c7, x)
        e4 = rpu.div(e4, y)

        e = rpu.add(e1, e2)
        e = rpu.add(e, e3)
        e = rpu.add(e, e4)

        n, d = rpu.frac(e)
        print(f"{n} / {d}")
    except ZeroByZeroDivisionError:
        assert (True)

    a = rpu.new(-5, 19)
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    print(bit_f.format(a))
    assert (a == 0b1110011101110111111111111111111111111111111111111111111111111111 or
            a == 0b1110011101100111111111111111111111111111111111111111111111111111)
    b = rpu.new(15, 27)
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    print(bit_f.format(b))
    assert (b == 0b0100110111111111111111111111111111111111111111111111111111111111 or
            b == 0b0100100111111111111111111111111111111111111111111111111111111111)
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (c == 0b0110010110111011101011111111111111111111111111111111111111111111 or
            c == 0b0110010110111011100011111111111111111111111111111111111111111111)
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    d = rpu.sub(a, b)
    print(bit_f.format(d))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(d, 0b1101101100111001100101011111111111111111111111111111111111111111))
    e = rpu.mul(a, b)
    print(bit_f.format(e))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(e, 0b1111001011011101110101111111111111111111111111111111111111111111))
    f = rpu.div(a, b)
    print(bit_f.format(f))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(f, 0b1110111101110111111111111111111111111111111111111111111111111111))

    a = rpu.new(3713, 28276)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(21946, 51272)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))

    a = rpu.new(-126, 122)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(-116, -34)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    d = rpu.new(-126 * -34 + -116 * 122, 122 * -34)
    print(bit_f.format(d))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(c, d))

    a = rpu.new(-69, 123)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(-53, 100)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    d = rpu.new(-4473, 4100)
    print(bit_f.format(d))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(c, d))

    a = rpu.new(-109, 0)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(84, -106)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    d = rpu.new(-109 * 106, 0)
    print(bit_f.format(d))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(c, d))

    a = rpu.new(7, 84)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(2, 127)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    d = rpu.new(1057, 10668)
    print(bit_f.format(d))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(c, d))

    a = rpu.new(59, 61)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(-97, 111)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    d = rpu.new(632, 6771)
    print(bit_f.format(d))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(c, d))

    a = rpu.new(113, 47)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(-118, 49)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    d = rpu.new(-9, 2303)
    print(bit_f.format(d))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(c, d))

    a = rpu.new(-32, 45)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(64, 91)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    d = rpu.new(-32, 4095)
    print(bit_f.format(d))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(c, d))

    a = rpu.new(1, 0)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(1, 0)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    d = rpu.new(1, 0)
    print(bit_f.format(d))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    assert (RPU.quickcmp(c, d))

    a = rpu.new(0, 1)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(1, 0)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    try:
        c = rpu.mul(a, b)
        assert (False)
    except ZeroByZeroDivisionError:
        assert (True)

    a = rpu.new(2, 103)
    print(bit_f.format(a))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    b = rpu.new(-114, 112)
    print(bit_f.format(b))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    d = rpu.new(-5759, 5768)
    print(bit_f.format(d))
    print(RPU.debugRenderer(c))
    print(RPU.debugRenderer(d))
    assert (not bool(rpu.flags & rpu.Flag.INVALID))
    # assert (RPU.quickcmp(c, d))

    a = rpu.new(2, 1)
    b = rpu.lft(a, 6, -13, 2, -5)
    c = rpu.new(1, 1)
    # assert (RPU.quickcmp(b, c))

    n_d_bits = 8

    # for i in range(1 << 20):
    #     n_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
    #     d_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
    #     try:
    #         a = rpu.new(n_0, d_0)
    #     except ZeroByZeroDivisionError:
    #         continue
    #
    #     gcd = math.gcd(n_0, d_0)
    #
    #     if gcd != 0:
    #         n_0, d_0 = n_0 / gcd, d_0 / gcd
    #
    #     n_1, d_1 = rpu.frac(a)
    #
    #     gcd = math.gcd(n_1, d_1)
    #
    #     if gcd != 0:
    #         n_1, d_1 = n_1 / gcd, d_1 / gcd
    #
    #     if d_0 < 0:
    #         n_0 = -n_0
    #         d_0 = -d_0
    #     if n_0 != n_1 or d_0 != d_1:
    #         print("%d: \n%d / %d != %d / %d\n" % (i, n_0, d_0, n_1, d_1))
    #     assert(n_0 == n_1 and d_0 == d_1)
    #     if not (i % (1 << 8)):
    #         print(i)

    assert (not RPU.quickcmp(RPU.bcl(0b00001001110010111101110111110100),
                             RPU.bcl(0b00001001110010111101110111110000)))
    assert (RPU.quickcmp(RPU.bcl(0b0111111111111111111111111111111111111111111111111111111111111111),
                         RPU.bcl(0b1111111111111111111111111111111111111111111111111111111111111111)))
    assert (RPU.quickcmp(RPU.bcl(0b1111111111111111111111111111111111111111111111111111111111111111),
                         RPU.bcl(0b0111111111111111111111111111111111111111111111111111111111111111)))

    a = rpu.new(57, 13)
    b = rpu.new(19, 27)
    c = rpu.div(a, b)

    for i in range(1 << 16):
        n_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        n_1 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_1 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        try:
            a = rpu.new(n_0, d_0)
        except ZeroByZeroDivisionError:
            assert (n_0 == 0 and d_0 == 0)
            continue
        try:
            b = rpu.new(n_1, d_1)
        except ZeroByZeroDivisionError:
            assert (n_1 == 0 and d_1 == 0)
            continue
        try:
            c = rpu.sub(a, b)
        except ZeroByZeroDivisionError:
            print(f"a = {n_0}/{d_0}, b = {n_1}/{d_1}")
            continue
        if d_0 < 0:
            n_0 = -n_0
            d_0 = -d_0
        if d_1 < 0:
            n_1 = -n_1
            d_1 = -d_1
        if d_0 != 0 or d_1 != 0:
            d = rpu.new(n_0 * d_1 - n_1 * d_0, d_0 * d_1)
        else:
            assert ((n_0 < 0) == (n_1 < 0))
            if n_0 < 0:
                d = rpu.new(-1, 0)
            else:
                d = rpu.new(1, 0)
        assert (not bool(rpu.flags & rpu.Flag.INVALID))
        if not RPU.quickcmp(c, d):
            print("%d: \n%d / %d = %s \n%d / %d = %s \n+ \n%s != \n%s\n" % (i, n_0, d_0, bit_f.format(a), n_1, d_1, bit_f.format(b), bit_f.format(d), bit_f.format(c)))
        assert (RPU.quickcmp(c, d))
        if not (i % (1 << 8)):
            print(i)

    for i in range(1 << 16):
        n_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        n_1 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_1 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        try:
            a = rpu.new(n_0, d_0)
        except ZeroByZeroDivisionError:
            assert (n_0 == 0 and d_0 == 0)
            continue
        try:
            b = rpu.new(n_1, d_1)
        except ZeroByZeroDivisionError:
            assert (n_1 == 0 and d_1 == 0)
            continue
        try:
            c = rpu.mul(a, b)
        except ZeroByZeroDivisionError:
            print(f"a = {n_0}/{d_0}, b = {n_1}/{d_1}")
            continue
        if d_0 < 0:
            n_0 = -n_0
            d_0 = -d_0
        if d_1 < 0:
            n_1 = -n_1
            d_1 = -d_1
        if d_0 != 0 and d_1 != 0:
            d = rpu.new(n_0 * n_1, d_0 * d_1)
        else:
            assert (n_0 != 0 and n_1 != 0)
            if (n_0 < 0) == (n_1 < 0):
                d = rpu.new(1, 0)
            else:
                d = rpu.new(-1, 0)
        assert (not bool(rpu.flags & rpu.Flag.INVALID))
        if not RPU.quickcmp(c, d):
            print("%d: \n%d / %d = %s \n%d / %d = %s \n+ \n%s != \n%s\n" % (i, n_0, d_0, bit_f.format(a), n_1, d_1, bit_f.format(b), bit_f.format(d), bit_f.format(c)))
        assert (RPU.quickcmp(c, d))
        if not (i % (1 << 8)):
            print(i)

    for i in range(1 << 16):
        n_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        n_1 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_1 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        try:
            a = rpu.new(n_0, d_0)
        except ZeroByZeroDivisionError:
            assert (n_0 == 0 and d_0 == 0)
            continue
        try:
            b = rpu.new(n_1, d_1)
        except ZeroByZeroDivisionError:
            assert (n_1 == 0 and d_1 == 0)
            continue
        try:
            c = rpu.div(a, b)
        except ZeroByZeroDivisionError:
            print(f"a = {n_0}/{d_0}, b = {n_1}/{d_1}")
            continue
        if d_0 < 0:
            n_0 = -n_0
            d_0 = -d_0
        if d_1 < 0:
            n_1 = -n_1
            d_1 = -d_1
        if d_0 != 0 and n_1 != 0:
            d = rpu.new(n_0 * d_1, d_0 * n_1)
        else:
            assert (n_0 != 0 and d_1 != 0)
            if n_0 > 0:
                d = rpu.new(1, 0)
            else:
                d = rpu.new(-1, 0)
        assert (not bool(rpu.flags & rpu.Flag.INVALID))
        if not RPU.quickcmp(c, d):
            print("%d: \n%d / %d = %s \n%d / %d = %s \n+ \n%s != \n%s\n" % (i, n_0, d_0, bit_f.format(a), n_1, d_1, bit_f.format(b), bit_f.format(d), bit_f.format(c)))
        assert (RPU.quickcmp(c, d))
        if not (i % (1 << 8)):
            print(i)

    for i in range(1 << 16):
        n_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        n_1 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_1 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        try:
            a = rpu.new(n_0, d_0)
        except ZeroByZeroDivisionError:
            assert (n_0 == 0 and d_0 == 0)
            continue
        try:
            b = rpu.new(n_1, d_1)
        except ZeroByZeroDivisionError:
            assert (n_1 == 0 and d_1 == 0)
            continue
        try:
            c = rpu.add(a, b)
        except ZeroByZeroDivisionError:
            print(f"a = {n_0}/{d_0}, b = {n_1}/{d_1}")
            continue
        if d_0 < 0:
            n_0 = -n_0
            d_0 = -d_0
        if d_1 < 0:
            n_1 = -n_1
            d_1 = -d_1
        if d_0 != 0 or d_1 != 0:
            d = rpu.new(n_0 * d_1 + n_1 * d_0, d_0 * d_1)
        else:
            assert ((n_0 < 0) == (n_1 < 0))
            if n_0 < 0:
                d = rpu.new(-1, 0)
            else:
                d = rpu.new(1, 0)
        assert (not bool(rpu.flags & rpu.Flag.INVALID))
        if not RPU.quickcmp(c, d):
            print("%d: \n%d / %d = %s \n%d / %d = %s \n+ \n%s != \n%s\n" % (i, n_0, d_0, bit_f.format(a), n_1, d_1, bit_f.format(b), bit_f.format(d), bit_f.format(c)))
        assert (RPU.quickcmp(c, d))
        if not (i % (1 << 8)):
            print(i)
