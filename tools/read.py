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
from me import VIS


# 读取五个通道的数据
def read_tiff_image(file_path):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    channel_data = band.ReadAsArray()
    return channel_data

# 读取rgb照片的数据
def read_rgb_image(file_path):
    rgb_image = Image.open(file_path)
    r, g, b = rgb_image.split()
    # 将通道数据转换为NumPy数组
    r_array = np.array(r)
    g_array = np.array(g)
    b_array = np.array(b)
    return r_array, g_array, b_array
