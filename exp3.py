import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread(r"picture/pepper.tif")
# 将图像转为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对图像进行平滑操作
smooth_img = cv2.blur(gray_img, (9, 9))
Gausi = cv2.GaussianBlur(gray_img, (9, 9), 1)  # 高斯滤波

# 将平滑后的图像转为uint8类型
smooth_img = smooth_img.astype('uint8')
Gausi = Gausi.astype('uint8')

plt.subplot(221)
plt.imshow(gray_img, cmap=plt.cm.gray)
plt.axis("off")
plt.title("src")
plt.subplot(222)
plt.imshow(smooth_img, cmap=plt.cm.gray)
plt.axis("off")
plt.title("tar")
plt.subplot(223)
plt.imshow(smooth_img, cmap=plt.cm.gray)
plt.axis("off")
plt.title("tar")
plt.subplot(224)
plt.imshow(Gausi, cmap=plt.cm.gray)
plt.axis("off")
plt.title("Gausi")
plt.show()
