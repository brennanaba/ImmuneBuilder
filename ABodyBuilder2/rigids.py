import torch

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.shape = x.shape
        assert (x.shape == y.shape) and (y.shape == z.shape), "x y and z should have the same shape"

    def __add__(self, vec):
        return Vector(vec.x + self.x, vec.y + self.y, vec.z + self.z)

    def __sub__(self, vec):
        return Vector(-vec.x + self.x, -vec.y + self.y, -vec.z + self.z)

    def __mul__(self, param):
        return Vector(param * self.x, param * self.y, param * self.z)

    def __matmul__(self, vec):
        return vec.x * self.x + vec.y * self.y + vec.z * self.z

    def norm(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2 + 1e-8) ** (1 / 2)

    def cross(self, other):
        a = (self.y * other.z - self.z * other.y)
        b = (self.z * other.x - self.x * other.z)
        c = (self.x * other.y - self.y * other.x)
        return Vector(a, b, c)

    def dist(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2 + 1e-8) ** (1 / 2)

    def unsqueeze(self, dim):
        return Vector(self.x.unsqueeze(dim), self.y.unsqueeze(dim), self.z.unsqueeze(dim))

    def squeeze(self, dim):
        return Vector(self.x.squeeze(dim), self.y.squeeze(dim), self.z.squeeze(dim))

    def map(self, func):
        return Vector(func(self.x), func(self.y), func(self.z))

    def to(self, device):
        return Vector(self.x.to(device), self.y.to(device), self.z.to(device))

    def __str__(self):
        return "Vector(x={},\ny={},\nz={})\n".format(self.x, self.y, self.z)

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return Vector(self.x[key], self.y[key], self.z[key])


class Rot:
    def __init__(self, xx, xy, xz, yx, yy, yz, zx, zy, zz):
        self.xx = xx
        self.xy = xy
        self.xz = xz
        self.yx = yx
        self.yy = yy
        self.yz = yz
        self.zx = zx
        self.zy = zy
        self.zz = zz
        self.shape = xx.shape

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                other.x * self.xx + other.y * self.xy + other.z * self.xz,
                other.x * self.yx + other.y * self.yy + other.z * self.yz,
                other.x * self.zx + other.y * self.zy + other.z * self.zz)

        if isinstance(other, Rot):
            return Rot(
                xx=self.xx * other.xx + self.xy * other.yx + self.xz * other.zx,
                xy=self.xx * other.xy + self.xy * other.yy + self.xz * other.zy,
                xz=self.xx * other.xz + self.xy * other.yz + self.xz * other.zz,
                yx=self.yx * other.xx + self.yy * other.yx + self.yz * other.zx,
                yy=self.yx * other.xy + self.yy * other.yy + self.yz * other.zy,
                yz=self.yx * other.xz + self.yy * other.yz + self.yz * other.zz,
                zx=self.zx * other.xx + self.zy * other.yx + self.zz * other.zx,
                zy=self.zx * other.xy + self.zy * other.yy + self.zz * other.zy,
                zz=self.zx * other.xz + self.zy * other.yz + self.zz * other.zz,
            )

        else:
            raise ValueError("Matmul against {}".format(type(other)))

    def inv(self):
        return Rot(
            xx=self.xx, xy=self.yx, xz=self.zx,
            yx=self.xy, yy=self.yy, yz=self.zy,
            zx=self.xz, zy=self.yz, zz=self.zz
        )

    def det(self):
        return self.xx * self.yy * self.zz + self.xy * self.yz * self.zx + self.yx * self.zy * self.xz - self.xz * self.yy * self.zx - self.xy * self.yx * self.zz - self.xx * self.zy * self.yz

    def unsqueeze(self, dim):
        return Rot(
            self.xx.unsqueeze(dim=dim), self.xy.unsqueeze(dim=dim), self.xz.unsqueeze(dim=dim),
            self.yx.unsqueeze(dim=dim), self.yy.unsqueeze(dim=dim), self.yz.unsqueeze(dim=dim),
            self.zx.unsqueeze(dim=dim), self.zy.unsqueeze(dim=dim), self.zz.unsqueeze(dim=dim)
        )

    def squeeze(self, dim):
        return Rot(
            self.xx.squeeze(dim=dim), self.xy.squeeze(dim=dim), self.xz.squeeze(dim=dim),
            self.yx.squeeze(dim=dim), self.yy.squeeze(dim=dim), self.yz.squeeze(dim=dim),
            self.zx.squeeze(dim=dim), self.zy.squeeze(dim=dim), self.zz.squeeze(dim=dim)
        )

    def detach(self):
        return Rot(
            self.xx.detach(), self.xy.detach(), self.xz.detach(),
            self.yx.detach(), self.yy.detach(), self.yz.detach(),
            self.zx.detach(), self.zy.detach(), self.zz.detach()
        )

    def to(self, device):
        return Rot(
            self.xx.to(device), self.xy.to(device), self.xz.to(device),
            self.yx.to(device), self.yy.to(device), self.yz.to(device),
            self.zx.to(device), self.zy.to(device), self.zz.to(device)
        )

    def __str__(self):
        return "Rot(xx={},\nxy={},\nxz={},\nyx={},\nyy={},\nyz={},\nzx={},\nzy={},\nzz={})\n".format(self.xx, self.xy,
                                                                                                     self.xz, self.yx,
                                                                                                     self.yy, self.yz,
                                                                                                     self.zx, self.zy,
                                                                                                     self.zz)

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return Rot(
            self.xx[key], self.xy[key], self.xz[key],
            self.yx[key], self.yy[key], self.yz[key],
            self.zx[key], self.zy[key], self.zz[key]
        )


class Rigid:
    def __init__(self, origin, rot):
        self.origin = origin
        self.rot = rot
        self.shape = self.origin.shape

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return self.rot @ other + self.origin
        if isinstance(other, Rigid):
            return Rigid(self.rot @ other.origin + self.origin, self.rot @ other.rot)

    def inv(self):
        inv_rot = self.rot.inv()
        t = inv_rot @ self.origin
        return Rigid(Vector(-t.x, -t.y, -t.z), inv_rot)

    def unsqueeze(self, dim=None):
        return Rigid(self.origin.unsqueeze(dim=dim), self.rot.unsqueeze(dim=dim))

    def squeeze(self, dim=None):
        return Rigid(self.origin.squeeze(dim=dim), self.rot.squeeze(dim=dim))

    def to(self, device):
        return Rigid(self.origin.to(device), self.rot.to(device))

    def __str__(self):
        return "Rigid(origin={},\nrot={})".format(self.origin, self.rot)

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return Rigid(self.origin[key], self.rot[key])


def vec_from_tensor(tens):
    assert tens.shape[-1] == 3, "What dimension you in?"
    return Vector(tens[..., 0], tens[..., 1], tens[..., 2])


def rigid_from_three_points(origin, y_x_plane, x_axis):
    v1 = x_axis - origin
    v2 = y_x_plane - origin

    v1 *= 1 / v1.norm()
    v2 = v2 - v1 * (v1 @ v2)
    v2 *= 1 / v2.norm()
    v3 = v1.cross(v2)
    rot = Rot(v1.x, v2.x, v3.x, v1.y, v2.y, v3.y, v1.z, v2.z, v3.z)
    return Rigid(origin, rot)


def rigid_from_tensor(tens):
    assert (tens.shape[-1] == 3), "I want 3D points"
    return rigid_from_three_points(vec_from_tensor(tens[..., 0, :]), vec_from_tensor(tens[..., 1, :]),
                                   vec_from_tensor(tens[..., 2, :]))

