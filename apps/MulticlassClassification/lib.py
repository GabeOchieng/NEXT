import numpy as np


def label_decode(l):
    out = 0
    for i, x in enumerate(l[::-1]):
        out += x*2**i
    return out

