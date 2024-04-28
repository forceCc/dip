import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('picture/pepper.tif', 0)
# flatten() 将数组变成一维
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
# 计算累积分布图
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

img = cv2.imread('picture/Lena.Bmp', 0)
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
# stacking images side-by-side
cv2.imshow('img', res)
cv2.waitKey()
cv2.destroyAllWindows()

# 彩色图像均衡化
img = cv2.imread('picture/pepper.tif', 1)
b, g, r = cv2.split(img)  # 通道分解
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH), )  # 通道合成
res = np.hstack((img, result))
cv2.imshow('dst', res)
cv2.waitKey(0)
