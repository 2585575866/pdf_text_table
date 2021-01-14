import os

import cv2
import numpy as np
import sys


import matplotlib.pyplot as plt

from excel_extract.excel_extract_java import FindContours, get_Affine_Location

'''水平投影'''


def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # 绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255
    cv2.imshow('hProjection2', hProjection)

    return h_


def getVProjection(image):
    vProjection = np.zeros(image.shape, np.uint8);
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0] * w
    # 循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    # 绘制垂直平投影图像
    for x in range(w):
        for y in range(h - w_[x], h):
            vProjection[y, x] = 255
    # cv2.imshow('vProjection',vProjection)
    return w_


def cut_png(png_dir,png_path,png_name,png_post):
    origineImage = cv2.imread(png_path)
    # 图像灰度化
    # image = cv2.imread('test.jpg',0)
    image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', image)
    # 将图片二值化
    retval, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('binary', img)
    # 图像高与宽
    (h, w) = img.shape
    Position = []
    # cv2.imshow('cropImg',cropImg)
    # 对行图像进行垂直投影
    W = getVProjection(img)
    print(W)
    Wstart = 0
    Wend = 0
    W_Start = 0
    W_End = 0
    for j in range(len(W)):
        if W[j] > 20 and Wstart == 0:
            W_Start = j
            Wstart = 1
            Wend = 0
        if W[j] < 20 and Wstart == 1:
            W_End = j
            Wstart = 0
            Wend = 1
        if Wend == 1 and W_End - W_Start >= 100:
            Position.append([W_Start, 0, W_End, h])
            Wend = 0
    # 根据确定的位置分割字符

    output_dir = png_dir + "/" + png_name
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    i = 0
    for m in range(len(Position)):
        # cv2.rectangle(origineImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 229, 238),1)
        new_image = origineImage[Position[m][1]:Position[m][3], Position[m][0]:Position[m][2]]
        new_image_shape = new_image.shape
        weight = new_image_shape[0]
        # print(new_image_shape)
        half_weight = int(weight / 2)
        # cv2.imwrite(output_dir + str(i) + "---.png", new_image)
        new_image1 = new_image[0:half_weight, :]
        new_image2 = new_image[half_weight:, :]
        # 保存到指定目录
        cv2.imwrite(output_dir + "/" + png_name + "_word_" + str(i) + "."+png_post, new_image1)
        i = i + 1
        cv2.imwrite(output_dir + "/" + png_name + "_word_" + str(i) + "."+png_post, new_image2)
        i = i + 1
        # cv2.imshow('image', new_image)
        cv2.waitKey(0)

def table_extract(input_dir,file_name,png_post):
    # input_Path = '../data/cut_output'
    output_excel_dir = input_dir

    # cutImg_name = input_Path.split('/')[-1][:-4]

    for root, dirs, files in os.walk(input_dir):
        if len(files) > 0:
            for file in files:
                img_input_path = os.path.join(root, file)
                src_img, Net_img, contours = FindContours(img_input_path)
                get_Affine_Location(src_img, Net_img, contours, file_name, output_excel_dir,png_post)


if __name__ == "__main__":

    #图片目录/root/tmp/
    png_dir = sys.argv[1]
    #图片名称   png1
    png_name = sys.argv[2]
    #图片后缀
    png_post = sys.argv[4]
    #图片路径
    png_path = png_dir+"/"+png_name+"."+png_post


    #图片切割
    cut_png(png_dir, png_path, png_name,png_post)
    #表格提取
    input_dir = png_dir+"/"+png_name
    table_extract(input_dir, png_name, png_post)





