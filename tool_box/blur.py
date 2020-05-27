import numpy as np
import math


def saltypepper(img, n):
    m = int((img.shape[0] * img.shape[1]) * n)
    for a in range(m):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        s = np.abs(np.random.normal(loc=0.0, scale=0.9))
        if s > 1.0:
            s = s * s
        s = s + 1.0
        if img.ndim == 2:
            img[j, i] = 1.0 * s
        elif img.ndim == 3:
            img[j, i, 0] = 1.0 * s
            img[j, i, 1] = 1.0 * s
            img[j, i, 2] = 1.0 * s
    return img


def motionblur(length, angle):
    len = max(1, length)
    EPS = np.finfo(float).eps
    half = (len - 1) / 2
    phi = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
    cosphi = math.cos(phi)
    sinphi = math.sin(phi)
    if cosphi < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1

    linewdt = 1
    sx = int(half * cosphi + linewdt * xsign - length * EPS)
    sy = int(half * sinphi + linewdt - length * EPS)
    if sx > 0:
        tx = 1
    else:
        tx = -1

    x, y = np.meshgrid(np.arange(0, sx + tx, tx), np.arange(0, sy + 1, 1))

    dist2line = (y * cosphi - x * sinphi)

    rad = np.sqrt(x ** 2 + y ** 2)

    lastpix = ((rad >= half) & (abs(dist2line) <= linewdt))
    x2lastpix = rad.copy()
    x2lastpix[lastpix] = half - np.abs((x[lastpix] + dist2line[lastpix] * sinphi) / cosphi)

    dist2line[lastpix] = np.sqrt(dist2line[lastpix] ** 2 + x2lastpix[lastpix] ** 2)
    dist2line = linewdt + EPS - np.abs(dist2line)
    dist2line[dist2line < 0] = 0

    h = np.zeros((sy + sy + 1, abs(sx + sx) + 1))
    h[0: sy + 1, 0:abs(sx) + 1] = np.flip(np.flip(dist2line, 0), 1)
    h[sy: sy + sy + 1, abs(sx):abs(sx + sx) + 1] = dist2line
    h = h / (h.sum() + EPS * len * len)
    if cosphi > 0:
        h = np.flip(h, 0)
    return h


def gaussianblur(siz, std):
    EPS = np.finfo(float).eps
    siz = (np.array(siz) - 1) / 2
    x, y = np.meshgrid(np.arange(- siz[1], siz[1] + 1, 1), np.arange(- siz[0], siz[0] + 1, 1))
    arg = - (x ** 2 + y ** 2) / (2 * std * std)
    h = np.exp(arg)
    h[h < EPS * h.max()] = 0

    if h.sum() != 0:
        h = h / h.sum()
    return h


if __name__ == '__main__':
    a = np.zeros([6, 6])
    b = saltypepper(a.copy(), 0.5)
    print(a)
    print(b)
    b = motionblur(25, 35)
    print(b)
    print(gaussianblur([3, 3], 0.5))
