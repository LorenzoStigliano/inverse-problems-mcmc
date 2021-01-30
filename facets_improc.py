#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

__all__ = \
    [
        "empty_true_color",
        "gaussian_filter",
        "get_bit_plane",
        "imhist",
        "imread",
        "imscale",
        "imshow",
        "imshow_reduced_color_depth",
        "imwrite",
        "linear_filter",
        "median_filter",
        "rank_filter",
        "zero_bit_plane"
    ]


def _img_a_cast(img_a, dtype, true_color=False):
    img_a = np.maximum(img_a, 0)
    img_a = np.minimum(img_a, 255)
    img_a = np.round(img_a, 0)
    img_a = np.array(img_a + 1.0e-6, dtype=dtype)
    if len(img_a.shape) == 2:
        if true_color:
            img_a_gs = np.zeros((img_a.shape[0], img_a.shape[1], 3),
                                dtype=dtype)
            for k in range(3):
                img_a_gs[:, :, k] = img_a
            return img_a_gs
        else:
            return img_a
    else:
        if len(img_a.shape) != 3 or img_a.shape[2] != 3:
            raise RuntimeError("Unexpected image type")
        return img_a


def imread(filename):
    img = PIL.Image.open(filename, "r")

    return np.array(img, dtype=np.int64)


def imwrite(filename, img_a, **kwargs):
    img_a = _img_a_cast(img_a, dtype=np.uint8)
    img = PIL.Image.fromarray(img_a)
    img.save(filename, **kwargs)


def imshow(img_a, new_figure=True):
    img_a = _img_a_cast(img_a, dtype=np.uint8, true_color=True)

    if new_figure:
        plt.figure()
    plt.imshow(img_a)
    plt.xticks([])
    plt.yticks([])


def imhist(img_a, new_figure=True):
    img_a = _img_a_cast(img_a, dtype=np.uint8)

    if len(img_a.shape) != 2:
        assert len(img_a.shape) == 3
        assert img_a.shape[2] == 3
        img_a = np.maximum(img_a[:, :, 0], img_a[:, :, 1], img_a[:, :, 2])

    if new_figure:
        plt.figure()
    plt.hist(img_a.flatten(), bins=256, range=(0, 256), color="k")
    plt.xlim(0, 256)


def imscale(img_a, factor, interpolation="nearest"):
    M, N = img_a.shape[:2]
    M_scaled = int(max(round(M * factor, 0), 1) + 1.0e-6)
    N_scaled = int(max(round(N * factor, 0), 1) + 1.0e-6)

    img_a = _img_a_cast(img_a, dtype=np.uint8)
    img = PIL.Image.fromarray(img_a)
    img = img.resize((M_scaled, N_scaled),
                     resample={"nearest": PIL.Image.NEAREST,
                               "bilinear": PIL.Image.BILINEAR,
                               "bicubic": PIL.Image.BICUBIC,
                               "lanczos": PIL.Image.LANCZOS}[interpolation])

    return np.array(img, dtype=np.int64)


def empty_true_color(N, M):
    return np.zeros((M, N, 3), dtype=np.int64)


def imshow_reduced_color_depth(img_a, d, new_figure=False):
    if d < 0 or d > 7:
        raise RuntimeError("Invalid color depth reduction")

    img_a = _img_a_cast(img_a, dtype=np.int64)
    img_a = np.right_shift(img_a, d)
    img_a = (255 * img_a) / (2 ** (8 - d) - 1)

    imshow(img_a, new_figure=new_figure)


def get_bit_plane(img_a, n):
    if n < 1 or n > 8:
        raise RuntimeError("Invalid bit plane")

    img_a = _img_a_cast(img_a, dtype=np.int64)
    img_a = np.bitwise_and(img_a, 2 ** (8 - n))
    img_a = 255 * np.right_shift(img_a, 8 - n)

    return _img_a_cast(img_a, dtype=np.int64)


def zero_bit_plane(img_a, n):
    if n < 1 or n > 8:
        raise RuntimeError("Invalid bit plane")

    img_a = _img_a_cast(img_a, dtype=np.int64)
    img_a = np.bitwise_and(img_a, 255 - 2 ** (8 - n))

    return _img_a_cast(img_a, dtype=np.int64)


def linear_filter(img_a, W, **kwargs):
    img_a = _img_a_cast(img_a, dtype=np.int64)
    W = np.fliplr(np.flipud(W))

    if len(img_a.shape) == 2:
        img_filtered_a = scipy.ndimage.convolve(img_a, W, **kwargs)
    else:
        assert len(img_a.shape) == 3
        assert img_a.shape[2] == 3
        img_filtered_a = np.zeros_like(img_a)
        for k in range(3):
            img_filtered_a[:, :, k] = scipy.ndimage.convolve(
                img_a[:, :, k], W, **kwargs)

    return _img_a_cast(img_filtered_a, dtype=np.int64)


def median_filter(img_a, size, **kwargs):
    img_a = _img_a_cast(img_a, dtype=np.int64)

    if len(img_a.shape) == 2:
        img_filtered_a = scipy.ndimage.median_filter(
            img_a, size=size, footprint=None, **kwargs)
    else:
        assert len(img_a.shape) == 3
        assert img_a.shape[2] == 3
        img_filtered_a = np.zeros_like(img_a)
        for k in range(3):
            img_filtered_a[:, :, k] = scipy.ndimage.median_filter(
                img_a[:, :, k], size=size, footprint=None, **kwargs)

    return _img_a_cast(img_filtered_a, dtype=np.int64)


def rank_filter(img_a, rank, size, **kwargs):
    img_a = _img_a_cast(img_a, dtype=np.int64)

    if len(img_a.shape) == 2:
        img_filtered_a = scipy.ndimage.rank_filter(
            img_a, rank, size=size, footprint=None, **kwargs)
    else:
        assert len(img_a.shape) == 3
        assert img_a.shape[2] == 3
        img_filtered_a = np.zeros_like(img_a)
        for k in range(3):
            img_filtered_a[:, :, k] = scipy.ndimage.rank_filter(
                img_a[:, :, k], rank, size=size, footprint=None, **kwargs)

    return _img_a_cast(img_filtered_a, dtype=np.int64)


def gaussian_filter(img_a, sigma, **kwargs):
    img_a = _img_a_cast(img_a, dtype=np.int64)

    if len(img_a.shape) == 2:
        img_filtered_a = scipy.ndimage.gaussian_filter(
            img_a, sigma, **kwargs)
    else:
        assert len(img_a.shape) == 3
        assert img_a.shape[2] == 3
        img_filtered_a = np.zeros_like(img_a)
        for k in range(3):
            img_filtered_a[:, :, k] = scipy.ndimage.gaussian_filter(
                img_a[:, :, k], sigma, **kwargs)

    return _img_a_cast(img_filtered_a, dtype=np.int64)
