import cv2
import numpy as np


def d_resize(x, d, fac=1.0):
    new_min = min(int(d[1] * fac), int(d[0] * fac))
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
    return y


def vis(region_map, color_map):
    color = d_resize(color_map, region_map.shape)
    indexs = (region_map.astype(np.float32)[:, :, 0] * 255 + region_map.astype(np.float32)[:, :, 1]) * 255 + region_map.astype(np.float32)[:, :, 2]
    result = np.zeros_like(color, dtype=np.uint8)
    for ids in [np.where(indexs == idsn) for idsn in np.unique(indexs).tolist()]:
        result[ids] = np.median(color[ids], axis=0)
    return result