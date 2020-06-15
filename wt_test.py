
import cv2
import numpy as np
from numpy import *
from pywt import dwt2, idwt2
import pywt
import matplotlib.pyplot as plt



# In[ ]:


# def method_1(image):
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     return binary
#
#
# def method_2(image):
#     blurred = cv.GaussianBlur(image, (3, 3), 0)  # 高斯去噪音
#     gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
#     t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     return binary
#
# def method_3_old(image):
#     blurred = cv.pyrMeanShiftFiltering(image, 100, 100)  # 先均值迁移去噪声
#     gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
#     t, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     return binary
#
#
# def method_3(image):
#     blurred = cv.pyrMeanShiftFiltering(image, 10, 100)  # 先均值迁移去噪声
#     gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
#     t, binary = cv.threshold(gray,0, 255, cv.THRESH_BINARY| cv.THRESH_OTSU)
#
#     contours, hierarch = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     for i in range (len(contours)):
#         area = cv.contourArea(contours[i])
#         if area < 200:
#             cv.drawContours(binary, [contours[i]], 0, 0, -1)
#     # binary = cv.cvtColor(binary,cv.COLOR_GRAY2RGB)
#     return binary

def wt_change():
    img = cv2.imread('./1600/1.jpg')

    # 对img进行haar小波变换：
    cA, (cH, cV, cD) = dwt2(img, 'haar')

    # 小波变换之后，低频分量对应的图像：
    cv2.imwrite('lena.png', np.uint8(cA / np.max(cA) * 255))
    # 小波变换之后，水平方向高频分量对应的图像：
    cv2.imwrite('lena_h.png', np.uint8(cH / np.max(cH) * 255))
    # 小波变换之后，垂直平方向高频分量对应的图像：
    cv2.imwrite('lena_v.png', np.uint8(cV / np.max(cV) * 255))
    # 小波变换之后，对角线方向高频分量对应的图像：
    cv2.imwrite('lena_d.png', np.uint8(cD / np.max(cD) * 255))

    # 根据小波系数重构回去的图像
    rimg = idwt2((cA, (cH, cV, cD)), 'haar')

def test_2():
    img = cv2.imread('./1600/1.jpg')
    img = cv2.resize(img, (448, 448))
    # 将多通道图像变为单通道图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    plt.figure('二维小波一级变换')
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs

    # 将各个子图进行拼接，最后得到一张图
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    img = np.concatenate([AH, VD], axis=0)

    # 显示为灰度图
    plt.imshow(img, 'gray')
    plt.title('result')
    plt.show()

if __name__ == '__main__':
    test_2()