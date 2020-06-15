#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np


# In[ ]:


def method_1(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary


def method_2(image):  
    blurred = cv.GaussianBlur(image, (3, 3), 0) # 高斯去噪音
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary


def method_3(image): 
    blurred = cv.pyrMeanShiftFiltering(image, 100, 100)  # 先均值迁移去噪声
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary


src = cv.imread("1600/1.jpg")
h, w = src.shape[:2]
ret = method_3(src)

result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h,0:w,:] = src
result[0:h,w:2*w,:] = cv.cvtColor(ret, cv.COLOR_GRAY2BGR)


#
# THRESH_BINARY = 0
# THRESH_BINARY_INV = 1
# THRESH_TRUNC = 2
# THRESH_TOZERO = 3
# THRESH_TOZERO_INV = 4
#
src = result
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
h, w = src.shape[:2]

# 自动阈值分割 OTSU
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
print("ret :", ret)

result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h,0:w,:] = src
result[0:h,w:2*w,:] = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

src = result

# 高斯模糊去噪声
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv.GaussianBlur(gray, (9, 9), 2, 2)
binary_1 = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY_INV, 45, 15)

# 开操作
se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))
binary = cv.morphologyEx(binary_1, cv.MORPH_OPEN, se)

#cv.imshow("binary", np.hstack((binary_1,gray,binary)))
cv.imshow("binary_1", binary_1)
cv.imshow("gray", gray)
cv.imshow("binary", binary)

cv.waitKey(0)
cv.destroyAllWindows()

