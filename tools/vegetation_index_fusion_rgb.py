import numpy as np
import os.path as osp
import tifffile as tf
from alive_progress import alive_bar
import os
import _thread
import threading
import time
from osgeo import gdal
from tqdm import tqdm
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

# 上面的是计算植被指数的方法

# 各个通道的数据位置
red_path = "E:/赤霉/choose/red_choose/"
green_path = "E:/赤霉/choose/green_choose/"
blue_path = "E:/赤霉/choose/blue_choose/"
red_edge_path = "E:/赤霉/choose/red_edge_choose/"
nir_path = "E:/赤霉/choose/nir_choose/"

# rgb照片的绝对路径
rgb_path = "E:/赤霉/choose/rgb_choose/"

# 输出位置 植被指数都有ndvi、psri、sipi、gli、rgr、ari
output_folder = "E:/赤霉/vegetation_index_fusion_rgb/ALL/ALL_1/"

# 检查是否存在文件夹，不存在创建
check_folder = "E:/赤霉/vegetation_index_fusion_rgb/ALL/ALL_1"

# 打开文件（如果不存在则创建新文件）
file = open(output_folder + 'introduce.txt', 'w')

# 写入内容
file.write('0.5*rgb_ph + 0.1*PSRI + 0.1*ARI + 0.1*SIPI + 0.1*RGR + 0.1*GLI')

# 关闭文件
file.close()

if not os.path.exists(check_folder):
    os.makedirs(check_folder)

# 读取顺序文件
order_file = 'E:/赤霉/choose/choose顺序.txt'
with open(order_file, 'r') as file:
    order = [line.strip() for line in file.readlines()]

rgb_names = []
names = []
for item in order:
    name = item + ".tif"
    names.append(name)
    rgb_name = item + ".jpg"
    rgb_names.append(rgb_name)

path = "E:/赤霉/choose/"


for rgb_name, name in tqdm(zip(rgb_names, names), desc='Processing'):
    # rgb照片的绝对路径
    rgb_file = rgb_path + rgb_name

    # tif格式照片的绝对路径，下面这个五个分别是单通道的图片，依次为红光、绿光、蓝光、近红外、红外。
    red_file = red_path + name
    green_file = green_path + name
    blue_file = blue_path + name
    nir_file = nir_path + name
    red_edge_file = red_edge_path + name
