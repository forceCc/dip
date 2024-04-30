import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读图
# img = cv2.imread(r'picture/pepper.tif')
# # 转换成灰度图
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 显示灰度图
# cv2.imshow('gray_img', img2)
# cv2.waitKey(0)
# # 获取直方图，由于灰度图img2是二维数组，需转换成一维数组
# plt.hist(img2.ravel(), 256)
# # 显示直方图
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# def calcGrayHist(I):
#     # 计算灰度直方图
#     h, w = I.shape[:2]
#     grayHist = np.zeros([256], np.uint64)
#     for i in range(h):
#         for j in range(w):
#             grayHist[I[i][j]] += 1
#     return grayHist
#
#
# img = cv2.imread(r'picture/pepper.tif')
# grayHist = calcGrayHist(img)
# x = np.arange(256)
# # 绘制灰度直方图
# plt.plot(x, grayHist, linewidth=2)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# img = cv2.imread(r'picture/pepper.tif', 0)
# # flatten() 将数组变成一维
# hist, bins = np.histogram(img.flatten(), 256, [0, 256])
# # 计算累积分布图
# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max() / cdf.max()
# plt.plot(cdf_normalized, color='b')
# plt.hist(img.flatten(), 256, [0, 256], color='r')
# plt.xlim([0, 256])
# plt.legend(('cdf', 'histogram'), loc='upper left')
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# img = cv2.imread(r'picture/Lena.Bmp', 0)
# equ = cv2.equalizeHist(img)
# res = np.hstack((img, equ))
# # stacking images side-by-side
# cv2.imshow('img', res)
# cv2.waitKey()
# cv2.destroyAllWindows()

img = cv2.imread(r'picture/pepper.tif', 1)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
axes[0].hist(img.ravel(), 256, [1, 256], color='blue')
b, g, r = cv2.split(img)  # 通道分解
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH), )  # 通道合成
axes[1].hist(result.ravel(), 256, [1, 256], color='red')
plt.show()
res = np.hstack((img, result))
cv2.imshow('dst', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 读取图片
img = cv2.imread('picture/pepper.tif', cv2.IMREAD_GRAYSCALE)

# 灰度转换
gray = img  # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 局部直方图均衡化处理
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))

# 将灰度图像和局部直方图相关联, 把直方图均衡化应用到灰度图
result = clahe.apply(gray)

# 显示图像
fig = plt.figure(figsize=(10, 10))  # 设置大小
plt.subplot(221)
plt.imshow(gray, cmap=plt.cm.gray, vmin=0, vmax=255), plt.axis("off"), plt.title('(a)')
plt.subplot(222)
plt.imshow(result, cmap=plt.cm.gray, vmin=0, vmax=255), plt.axis("off"), plt.title('(b)')
plt.subplot(223)
plt.hist(img.ravel(), 256), plt.title('(c)')
plt.subplot(224)
plt.hist(result.ravel(), 256), plt.title('(d)')
plt.show()
