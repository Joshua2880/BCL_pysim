import random
from typing import NewType
from enum import Enum
import math


class ZeroByZeroDivisionError(ArithmeticError):
    pass


class RPU:
    bcl = NewType('bcl', int)

    bcl_width = 64
    msb_mask = 1 << (bcl_width - 1)

    class Op(Enum):
        NOP = 0
        NEW = 1
        FRAC = 2
        LFT = 3
        BLFT = 4

    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0
        self.f = 0
        self.g = 0
        self.h = 0

        self.x_cntr = 0
        self.y_cntr = 0
        self.z_cntr = 0

        self.emission_ready = False
        self.emission_val = False

        self.op = RPU.Op.NOP

    def set_registers(self, a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.h = h

    def init_cntrs(self):
        self.x_cntr = 0
        self.y_cntr = 0
        self.z_cntr = 0

    # combinational logic in hardware version
    def update_state(self):
        n_0 = n_1 = n_2 = n_3 = 0
        d_0 = d_1 = d_2 = d_3 = 0

        if self.op == RPU.Op.NEW:
            n_0 = self.a
            d_0 = self.e
        elif self.op == RPU.Op.LFT:
            if self.x_cntr == 1:
                n_0, n_1 = self.a, self.b
                d_0, d_1 = self.e, self.f
            elif self.x_cntr > 2:
                n_0, n_1 = self.a, self.a + self.b
                d_0, d_1 = self.e, self.e + self.f
        elif self.op == RPU.Op.BLFT:
            if self.x_cntr == 1 and self.y_cntr == 1:
                n_0, n_1, n_2, n_3 = self.a, self.b, self.c, self.d
                d_0, d_1, d_2, d_3 = self.e, self.f, self.g, self.h
            elif self.x_cntr >= 2 and self.y_cntr >= 2:
                n_0, n_1, n_2, n_3 = self.a, self.a + self.b, self.a + self.c, self.a + self.b + self.c + self.d
                d_0, d_1, d_2, d_3 = self.e, self.e + self.f, self.e + self.g, self.e + self.f + self.g + self.h

        b_0 = b_1 = b_2 = b_3 = False

        if self.z_cntr == 0:
            b_0 = bool(n_0 & RPU.msb_mask) != bool(d_0 & RPU.msb_mask) and n_0 != 0
            b_1 = bool(n_1 & RPU.msb_mask) != bool(d_1 & RPU.msb_mask) and n_1 != 0
            b_2 = bool(n_2 & RPU.msb_mask) != bool(d_2 & RPU.msb_mask) and n_2 != 0
            b_3 = bool(n_3 & RPU.msb_mask) != bool(d_3 & RPU.msb_mask) and n_3 != 0
        elif self.z_cntr == 1:
            b_0 = 0 <= n_0 < d_0
            b_1 = 0 <= n_1 < d_1
            b_2 = 0 <= n_2 < d_2
            b_3 = 0 <= n_3 < d_3
        else:
            b_0 = (d_0 << 1) <= n_0
            b_1 = (d_1 << 1) <= n_1
            b_2 = (d_2 << 1) <= n_2
            b_3 = (d_3 << 1) <= n_3

        self.emission_ready = (self.op == RPU.Op.NEW or
                               self.op == RPU.Op.LFT and (b_0 == b_1 or self.x_cntr == RPU.bcl_width) or
                               self.op == RPU.Op.BLFT and (b_0 == b_1 == b_2 == b_3 or self.x_cntr == self.y_cntr == RPU.bcl_width)
                               ) and not self.z_cntr >= RPU.bcl_width
        self.emission_val = b_0

    def new(self, n: int, d: int) -> bcl:
        if n == 0 and d == 0:
            raise ZeroByZeroDivisionError
        self.op = RPU.Op.NEW
        self.set_registers(n, 0, 0, 0, d, 0, 0, 0)
        self.init_cntrs()

        z = RPU.bcl(0)

        self.update_state()
        while self.emission_ready:
            z = self.blft_emmit_z(z)
            self.update_state()

        return z

    def frac(self, x: bcl) -> (int, int):
        self.op = RPU.Op.FRAC
        self.set_registers(1, 0, 0, 0, 0, 0, 1, 0)
        self.init_cntrs()

        while self.x_cntr < RPU.bcl_width:
            x = self.blft_ingest_x(x)
        return self.a, self.e

    def blft(self, x: bcl, y: bcl, a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int) -> bcl:
        self.op = RPU.Op.BLFT
        self.set_registers(a, b, c, d, e, f, g, h)
        self.init_cntrs()

        z = RPU.bcl(0)

        while self.z_cntr < RPU.bcl_width:
            x = self.blft_ingest_x(x)
            y = self.blft_ingest_y(y)

            self.update_state()
            while self.emission_ready:
                z = self.blft_emmit_z(z)
                self.update_state()

        return z

    def normalize_sign(self):
        if self.e < 0:
            self.a, self.b, self.c, self.d, \
                self.e, self.f, self.g, self.h = \
                -self.a, -self.b, -self.c, -self.d, \
                    -self.e, -self.f, -self.g, -self.h

    def blft_ingest_x(self, x: bcl) -> bcl:
        if self.x_cntr == 0:
            if x & RPU.msb_mask:
                self.a, self.b, self.c, self.d, \
                    self.e, self.f, self.g, self.h = \
                    -self.a, -self.b, self.c, self.d, \
                        -self.e, -self.f, self.g, self.h
        elif self.x_cntr == 1:
            if x & RPU.msb_mask:
                self.a, self.b, self.c, self.d, \
                    self.e, self.f, self.g, self.h = \
                    self.c, self.d, self.a, self.b, \
                        self.g, self.h, self.e, self.f
        elif self.x_cntr < RPU.bcl_width:
            if x & RPU.msb_mask:
                if self.c & 1 or self.d & 1 or self.g & 1 or self.h & 1:
                    self.a, self.b, self.c, self.d, \
                        self.e, self.f, self.g, self.h = \
                        self.a << 1, self.b << 1, self.c, self.d, \
                            self.e << 1, self.f << 1, self.g, self.h
                else:
                    self.a, self.b, self.c, self.d, \
                        self.e, self.f, self.g, self.h = \
                        self.a, self.b, self.c >> 1, self.d >> 1, \
                        self.e, self.f, self.g >> 1, self.h >> 1
            else:
                self.a, self.b, self.c, self.d, \
                    self.e, self.f, self.g, self.h = \
                    self.a + self.c, self.b + self.d, self.a, self.b, \
                        self.e + self.g, self.f + self.h, self.e, self.f

        self.normalize_sign()

        self.x_cntr += 1
        return RPU.bcl(x << 1)

    def blft_ingest_y(self, y: bcl) -> bcl:
        if self.y_cntr == 0:
            if y & RPU.msb_mask:
                self.a, self.b, self.c, self.d, \
                    self.e, self.f, self.g, self.h = \
                    -self.a, self.b, -self.c, self.d, \
                        -self.e, self.f, -self.g, self.h
        elif self.y_cntr == 1:
            if y & RPU.msb_mask:
                self.a, self.b, self.c, self.d, \
                    self.e, self.f, self.g, self.h = \
                    self.b, self.a, self.d, self.c, \
                        self.f, self.e, self.h, self.g
        elif self.y_cntr < RPU.bcl_width:
            if y & RPU.msb_mask:
                if self.b & 1 or self.d & 1 or self.f & 1 or self.h & 1:
                    self.a, self.b, self.c, self.d, \
                        self.e, self.f, self.g, self.h = \
                        self.a << 1, self.b, self.c << 1, self.d, \
                            self.e << 1, self.f, self.g << 1, self.h
                else:
                    self.a, self.b, self.c, self.d, \
                        self.e, self.f, self.g, self.h = \
                        self.a, self.b >> 1, self.c, self.d >> 1, \
                        self.e, self.f >> 1, self.g, self.h >> 1
            else:
                self.a, self.b, self.c, self.d, \
                    self.e, self.f, self.g, self.h = \
                    self.a + self.b, self.a, self.c + self.d, self.c, \
                        self.e + self.f, self.e, self.g + self.h, self.g

        self.normalize_sign()

        self.y_cntr += 1
        return RPU.bcl(y << 1)

    def blft_emmit_z(self, z: bcl) -> bcl:
        z <<= 1
        z |= self.emission_val

        if self.z_cntr == 0:
            if self.emission_val:
                self.a, self.b, self.c, self.d, \
                    self.e, self.f, self.g, self.h = \
                    -self.a, -self.b, -self.c, -self.d, \
                        self.e, self.f, self.g, self.h
        elif self.z_cntr == 1:
            if self.emission_val:
                self.a, self.b, self.c, self.d, \
                    self.e, self.f, self.g, self.h = \
                    self.e, self.f, self.g, self.h, \
                        self.a, self.b, self.c, self.d
        else:
            if self.emission_val:
                if self.a & 1 or self.b & 1 or self.c & 1 or self.d & 1:
                    self.a, self.b, self.c, self.d, \
                        self.e, self.f, self.g, self.h = \
                        self.a, self.b, self.c, self.d, \
                            self.e << 1, self.f << 1, self.g << 1, self.h << 1
                else:
                    self.a, self.b, self.c, self.d, \
                        self.e, self.f, self.g, self.h = \
                        self.a >> 1, self.b >> 1, self.c >> 1, self.d >> 1, \
                            self.e, self.f, self.g, self.h
            else:
                self.a, self.b, self.c, self.d, \
                    self.e, self.f, self.g, self.h = \
                    self.e, self.f, self.g, self.h, \
                        self.a - self.e, self.b - self.f, self.c - self.g, self.d - self.h

        self.normalize_sign()

        self.z_cntr += 1
        return RPU.bcl(z)

    def add(self, x: bcl, y: bcl) -> bcl:
        return self.blft(x, y, 0, 1, 1, 0, 0, 0, 0, 1)

    def sub(self, x: bcl, y: bcl) -> bcl:
        return self.blft(x, y, 0, 1, -1, 0, 0, 0, 0, 1)

    def mul(self, x: bcl, y: bcl) -> bcl:
        return self.blft(x, y, 1, 0, 0, 0, 0, 0, 0, 1)

    def div(self, x: bcl, y: bcl) -> bcl:
        return self.blft(x, y, 0, 1, 0, 0, 0, 0, 1, 0)


if __name__ == "__main__":
    bit_f = "{0:0%db}" % RPU.bcl_width
    rpu = RPU()
    a = rpu.new(-5, 19)
    print(bit_f.format(a))
    assert (a == 0b1110011101110111111111111111111111111111111111111111111111111111)
    b = rpu.new(15, 27)
    print(bit_f.format(b))
    assert (b == 0b0100110111111111111111111111111111111111111111111111111111111111)
    c = rpu.add(a, b)
    print(bit_f.format(c))
    assert (c == 0b0110010110111011101011111111111111111111111111111111111111111111)
    d = rpu.sub(a, b)
    print(bit_f.format(d))
    assert (d == 0b1101101100111001100101011111111111111111111111111111111111111111)
    e = rpu.mul(a, b)
    print(bit_f.format(e))
    assert (e == 0b1111001011011101110101111111111111111111111111111111111111111111)
    f = rpu.div(a, b)
    print(bit_f.format(f))
    assert (f == 0b1110111101110111111111111111111111111111111111111111111111111111)

    a = rpu.new(3713, 28276)
    print(bit_f.format(a))
    b = rpu.new(21946, 51272)
    print(bit_f.format(b))
    c = rpu.add(a, b)
    print(bit_f.format(c))

    a = rpu.new(-126, 122)
    print(bit_f.format(a))
    b = rpu.new(-116, -34)
    print(bit_f.format(b))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    d = rpu.new(-126 * -34 + -116 * 122, 122 * -34)
    print(bit_f.format(d))

    a = rpu.new(-69, 123)
    print(bit_f.format(a))
    b = rpu.new(-53, 100)
    print(bit_f.format(b))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    d = rpu.new(-4473, 4100)
    print(bit_f.format(d))

    a = rpu.new(-109, 0)
    print(bit_f.format(a))
    b = rpu.new(84, -106)
    print(bit_f.format(b))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    d = rpu.new(-109 * 106, 0)
    print(bit_f.format(d))

    a = rpu.new(7, 84)
    print(bit_f.format(a))
    b = rpu.new(2, 127)
    print(bit_f.format(b))
    c = rpu.add(a, b)
    print(bit_f.format(c))
    d = rpu.new(1057, 10668)
    print(bit_f.format(d))

    n_d_bits = 8

    for i in range(1 << 16):
        n_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        try:
            a = rpu.new(n_0, d_0)
        except ZeroByZeroDivisionError:
            continue

        gcd = math.gcd(n_0, d_0)

        if gcd != 0:
            n_0, d_0 = n_0 / gcd, d_0 / gcd

        n_1, d_1 = rpu.frac(a)

        gcd = math.gcd(n_1, d_1)

        if gcd != 0:
            n_1, d_1 = n_1 / gcd, d_1 / gcd

        if d_0 < 0:
            n_0 = -n_0
            d_0 = -d_0
        if n_0 != n_1 or d_0 != d_1:
            print("%d: \n%d / %d != %d / %d\n" % (i, n_0, d_0, n_1, d_1))
        assert(n_0 == n_1 and d_0 == d_1)

    for i in range(1 << 16):
        n_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_0 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        n_1 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        d_1 = random.randint(-(1 << (n_d_bits - 1)), 1 << (n_d_bits - 1))
        try:
            a = rpu.new(n_0, d_0)
        except ZeroByZeroDivisionError:
            continue
        try:
            b = rpu.new(n_1, d_1)
        except ZeroByZeroDivisionError:
            continue
        c = rpu.add(a, b)
        if d_0 < 0:
            n_0 = -n_0
            d_0 = -d_0
        if d_1 < 0:
            n_1 = -n_1
            d_1 = -d_1
        d = rpu.new(n_0 * d_1 + n_1 * d_0, d_0 * d_1)
        if c != d:
            print("%d: \n%d / %d = %s \n%d / %d = %s \n+ \n%s != \n%s\n" % (i, n_0, d_0, bit_f.format(a), n_1, d_1, bit_f.format(b), bit_f.format(d), bit_f.format(c)))
        assert(c == d)
