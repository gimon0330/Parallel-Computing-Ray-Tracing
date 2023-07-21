import numpy as np
from math import sqrt, acos, tan

def get_box(i, objects, camera, screen, width, height):
    ratio = float(width) / height

    x, y, z = objects[i]['center']
    a = np.linalg.norm(objects[i]['center'] - camera)
    ay = np.linalg.norm(np.array([0, y, z]) - camera)
    ax = np.linalg.norm(np.array([x, 0, z]) - camera)
    r = objects[i]['radius']

    def rescale(inp, mode):
        if mode == 'x':
            ret = round(inp * (width / 2) + (width / 2))
            ret = max(0, ret)
            ret = min(ret, width - 1)
            return ret
        elif mode == 'y':
            ret = round(inp * (height / 2 * ratio) + (height / 2))
            ret = max(0, ret)
            ret = min(ret, height - 1)
            return ret
        else:
            raise

    def method1():
        # x1, x2
        b = sqrt(abs(ax * ax - r * r))
        st = (abs(z) + 1) / ax
        if x == 0:
            x1 = -r / b
            x2 = r / b
        else:
            tt = st / sqrt(1 - st * st)
            x1 = (b - r * tt) / (b * tt + r)
            x2 = x1 + 2 * r * b / (ax * ax * st * st - r * r)
        sign = 1 if x >= 0 else -1
        x1 *= sign
        x2 *= sign
        x1, x2 = rescale(x1, 'x'), rescale(x2, 'x')
        x1, x2 = min(x1, x2), max(x1, x2)

        # y1, y2
        b = sqrt(abs(ay * ay - r * r))
        st = (abs(z) + 1) / ay
        if y == 0:
            y1 = -r / b
            y2 = r / b
        else:
            tt = st / sqrt(1 - st * st)
            y1 = (b - r * tt) / (b * tt + r)
            y2 = y1 + 2 * r * b / (ay * ay * st * st - r * r)
        sign = -1 if y >= 0 else 1
        y1 *= sign
        y2 *= sign
        y1, y2 = rescale(y1, 'y'), rescale(y2, 'y')
        y1, y2 = min(y1, y2), max(y1, y2)

        return x1, x2, y1, y2

    def method2():
        st = (abs(z) + 1) / a
        ct = sqrt(1 - st * st)
        tt = st / ct
        cphi = r / a
        sphi = sqrt(1 - cphi * cphi)
        tphi = sphi / cphi
        sxy = sqrt(x * x + y * y)

        l = (tphi - tt) / (1 + tphi * tt)
        pts = []
        rects = [(screen[0], screen[1]), (screen[0], screen[3]), (screen[2], screen[1]),
                 (screen[2], screen[3])]
        for (_x, _y) in rects:
            if x * _x + y * _y >= l * sxy:
                pts.append((_x, _y))
        if y != 0:
            for _x in [screen[0], screen[2]]:
                cand_y = (l * sxy - x * _x) / y
                if abs(cand_y) <= 1 / ratio:
                    pts.append((_x, cand_y))
        if x != 0:
            for _y in [screen[1], screen[3]]:
                cand_x = (l * sxy - y * _y) / x
                if abs(cand_x) <= 1:
                    pts.append((cand_x, _y))
        pts = np.array(pts)

        pts_x, pts_y = pts[:, 0], -pts[:, 1]
        return rescale(pts_x.min(), 'x'), rescale(pts_x.max(), 'x'), \
            rescale(pts_y.min(), 'y'), rescale(pts_x.max(), 'y')

    if ax >= r and ay >= r:
        return method1()
    else:
        return method2()