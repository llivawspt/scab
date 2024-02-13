import numpy as np
import os
from osgeo import gdal
from PIL import Image
import cv2


def psri(red, blue, nir):
    """
    PSRI：(red - blue) / nir
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        PSRI = np.true_divide((red - blue), nir)
        PSRI[~np.isfinite(PSRI)] = 0  # 处理除零错误
    return PSRI


def rgr(nir, green):
    """
    RGR：(nir - green)
    """
    RGR = nir - green
    return RGR


def sipi(nir, blue, red):
    """
    SIPI：(nir - blue) / (nir - red)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        SIPI = np.true_divide((nir - blue), (nir - red))
        SIPI[~np.isfinite(SIPI)] = 0  # 处理除零错误
    return SIPI


def ari(green, red):
    """
    ARI：(1 / green - 1 / red)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ARI = np.true_divide(1, green) - np.true_divide(1, red)
        ARI[~np.isfinite(ARI)] = 0  # 处理除零错误
    return ARI


def gli(green, red, blue):
    """
    GLI：(2*green-red-blue)/(2*green+red+blue)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        GLI = np.true_divide((2 * green - red - blue), (2 * green + red + blue))
        GLI[~np.isfinite(GLI)] = 0  # 处理除零错误
    return GLI


def ndvi(nir, red):
    """
    NIR = (nir - red) / (nir + red)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        NDVI = np.true_divide((nir - red), (nir + red))
        NDVI[~np.isfinite(NDVI)] = 0  # 处理除零错误
    return NDVI
