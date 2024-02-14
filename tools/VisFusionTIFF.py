import numpy as np
from labelme import utils
import os.path as osp
import tifffile as tf
from alive_progress import alive_bar
import os
import _thread
import threading
import time
from osgeo import gdal
from tqdm import tqdm

def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    # im_bands = dataset.RasterCount  # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    del dataset  # 关闭对象dataset，释放内存
    # return im_width, im_height, im_proj, im_geotrans, im_data,im_bands
    return im_proj, im_geotrans, im_data, im_width, im_height


def write_img(filename, im_proj, im_geotrans, im_data):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        # 注意数据的存储波段顺序：im_bands, im_height, im_width
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape  # 没看懂
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


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


# 各个通道的数据位置
red_path = "E:/赤霉/choose/red_choose/"
green_path = "E:/赤霉/choose/green_choose/"
blue_path = "E:/赤霉/choose/blue_choose/"
red_edge_path = "E:/赤霉/choose/red_edge_choose/"
nir_path = "E:/赤霉/choose/nir_choose/"

# tif数据位置
tif_path = "E:/赤霉/choose/four_channels_choose/"

# 输出位置 植被指数都有ndvi、psri、sipi、gli、rgr、ari
output_folder = "E:/赤霉/vegetation_index_fusion/ALL/ALL_1/"

# 检查是否存在文件夹，不存在创建
check_folder = "E:/赤霉/vegetation_index_fusion/ALL/ALL_1"

if not os.path.exists(check_folder):
    os.makedirs(check_folder)

# 读取顺序文件
order_file = 'E:/赤霉/choose/choose顺序.txt'
with open(order_file, 'r') as file:
    order = [line.strip() for line in file.readlines()]

names = []
for item in order:
    name = item + ".tif"
    names.append(name)

path = "E:/赤霉/choose/"

for name in tqdm(names, desc='Processing'):
    proj, geotrans, data1, row1, column1 = read_img(red_path + name)  # 红波段
    proj, geotrans, data2, row2, column2 = read_img(blue_path + name)  # 蓝光波段
    proj, geotrans, data3, row3, column3 = read_img(green_path + name)  # 绿光波段
    proj, geotrans, data4, row4, column4 = read_img(nir_path + name)  # 红外波段
    proj, geotrans, data5, row5, column5 = read_img(red_edge_path + name)  # 近红外波段

    proj, geotrans, tif, row6, column6 = read_img(tif_path + name)  # 四通道融合后的照片

    # red  = tf.imread(red_path + name)
    # blue = tf.imread(blue_path + name)
    # green = tf.imread(green_path + name)
    # nir = tf.imread(nir_path + name)
    # red_edge = tf.imread(red_edge_path + name)

    # ---------------------------------#
    # 计算出对应的植被指数                #
    # --------------------------------#
    NDVI = ndvi(data4, data1)
    PSRI = psri(data1, data2, data4)
    ARI = ari(data3, data1)
    SIPI = sipi(data4, data2, data1)
    RGR = rgr(data4, data3)
    GLI = gli(data3, data1, data2)

    # 波段顺序blue、green、red、nir

    percent = 0.5

    data = np.array(0.5*tif + 0.1*PSRI + 0.1*ARI + 0.1*SIPI + 0.1*RGR + 0.1*GLI, dtype=data1.dtype)

    output_file = osp.join(output_folder, name)


    write_img(output_file, proj, geotrans, data)
