import os

import cv2
import numpy as np

import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    # 读入原始图像
    root_dir = "../data/input"
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root,file)
            origineImage = cv2.imread(file_path)
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
                if W[j] > 15 and Wstart == 0:
                    W_Start = j
                    Wstart = 1
                    Wend = 0
                if W[j] < 15 and Wstart == 1:
                    W_End = j
                    Wstart = 0
                    Wend = 1
                if Wend == 1 and W_End - W_Start >=100:
                    Position.append([W_Start, 0, W_End, h])
                    Wend = 0
            # 根据确定的位置分割字符
            file_split=file.split(".")
            pre_file = file_split[0]
            output_dir = "../data/cut_output/"+pre_file+"/"
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            for m in range(len(Position)):
                # cv2.rectangle(origineImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 229, 238),1)
                new_image = origineImage[Position[m][1]:Position[m][3],Position[m][0]:Position[m][2]]
                #保存到指定目录
                cv2.imwrite(output_dir + str(m) + ".png", new_image)
                # cv2.imshow('image', new_image)
                cv2.waitKey(0)
