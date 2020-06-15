#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
from numpy import *


# In[ ]:


def method_1(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary


def method_2(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)  # 高斯去噪音
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary

def method_3_old(image):
    blurred = cv.pyrMeanShiftFiltering(image, 100, 100)  # 先均值迁移去噪声
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary


def method_3(image):
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)  # 先均值迁移去噪声
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray,0, 255, cv.THRESH_BINARY| cv.THRESH_OTSU)

    contours, hierarch = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for i in range (len(contours)):
        area = cv.contourArea(contours[i])
        if area < 200:
            cv.drawContours(binary, [contours[i]], 0, 0, -1)
    # binary = cv.cvtColor(binary,cv.COLOR_GRAY2RGB)
    return binary


def cal_x_density(lsit_x):
    pass

def remove_spot(img):
    Laplacian = cv.Laplacian(img, cv.CV_64F)
    Laplacian = cv.convertScaleAbs(Laplacian)

    _, labels, stats, centroids = cv.connectedComponentsWithStats(Laplacian)
    print(centroids)
    print("123")
    print("stats = ", stats)
    i = 0
    for istat in stats:
        if istat[4] < 10500:
            # print(i)
            print(istat[0:2])
            if istat[3] > istat[4]:
                r = istat[3]
            else:
                r = istat[4]
            cv.rectangle(Laplacian, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), (0, 0, 255), thickness=-1)
        i = i + 1

    cv.imwrite('105.jpg', Laplacian)
    return Laplacian

if __name__ == '__main__':

    src = cv.imread("1600/3.jpg")
    # h, w = src.shape[:2]
    # src = src[:,:,0]
    ret = method_3(src)
    print("ret.shape")
    print(ret.shape)
    # ret_s = np.zeros([h, w], dtype=src.dtype)


    # ret = cv.pyrMeanShiftFiltering(ret_s,10,100)
    # ret = method_2(src)
    # ret = method_1(src)

    # result = np.zeros([h, w * 2,3], dtype=src.dtype)
    # result[0:h, 0:w,:] = src
    # # result[0:h, w:2 * w, :] = cv.cvtColor(ret, cv.COLOR_GRAY2RGB)
    # result[0:h, w:2 * w,:] = ret
    #
    # result2 = np.zeros([h, w, 3], dtype=src.dtype)
    # result2 = cv.cvtColor(ret, cv.COLOR_GRAY2BGR)
    # print("result2.shape:")
    # np.set_printoptions(threshold=np.inf)
    #
    # print(result.shape)
    # single_channel_src = result[:,w:2*w]
    # width = single_channel_src.shape[1]

    # single_channel_src[]
    # ret = remove_spot(ret)

    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", ret)
    cv.imshow("src",src)

    cv.waitKey(0)
    cv.destroyAllWindows()
