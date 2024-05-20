import cv2
import numpy as np
from matplotlib import pyplot as plt


# Perwitt算子
def prewitt(img):
    r, c = img.shape[:2]
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    dx = np.array([[0, 0, -1], [1, 0, -1], [1, 0, -1]])
    dy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    for i in range(r):
        for j in range(c):
            if (j + 3 <= c) and (i + 3 <= r):
                new_imageX[i, j] = (np.sum(img[i : i + 3, j : j + 3] * dx)) ** 2
                new_imageX[i, j] = (np.sum(img[i : i + 3, j : j + 3] * dy)) ** 2
                new_image[i, j] = (
                    new_imageX[i, j] * new_imageX[i, j]
                    + new_imageY[i, j] * new_imageY[i, j]
                ) ** 0.5
    return np.uint8(new_image)


# robert算子
def robert1(img):
    h, w = img.shape[:2]
    r = [[-1, -1], [1, 1]]
    for i in range(h):
        for j in range(w):
            if (j + 2 < w) and (i + 2 <= h):
                process_img = img[i : i + 2, j : j + 2]
                list_robert = r * process_img
                img[i, j] = abs(list_robert.sum())
    return img


def robert2(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros((r, c))
    new_imageY = np.zeros((r, c))
    dx = [[-1, 0], [0, 1]]
    dy = [[0, -1], [1, 0]]
    for i in range(r):
        for j in range(c):
            if (j + 2 <= c) and (i + 2 <= r):
                new_imageX[i, j] = (np.sum(img[i : i + 2, j : j + 2] * dx)) ** 2
                new_imageX[i, j] = (np.sum(img[i : i + 2, j : j + 2] * dy)) ** 2
                new_image[i, j] = (
                    new_imageX[i, j] * new_imageX[i, j]
                    + new_imageY[i, j] * new_imageY[i, j]
                ) ** 0.5
    return np.uint8(new_image)


# Sobel算子
def sobel(img):
    h, w = img.shape
    new_img = np.zeros([h, w])
    x_img = np.zeros(img.shape)
    y_img = np.zeros(img.shape)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in range(h - 2):
        for j in range(w - 2):
            x_img[i + 1, j + 1] = abs(np.sum(img[i : i + 3, j : j + 3] * sobel_x))
            y_img[i + 1, j + 1] = abs(np.sum(img[i : i + 3, j : j + 3] * sobel_y))
            new_img[i + 1, j + 1] = np.sqrt(
                np.square(x_img[i + 1, j + 1]) + np.square(y_img[i + 1, j + 1])
            )
    return np.uint8(new_img)


# 常用的Laplace算子模板 [[0,1,0],[1,-4,1],[0,1,0]] [[1,1,1],[1,-8,1],[1,1,1]]
def laplace(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    L_sunnzi = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])

    for i in range(r - 3):
        for j in range(c - 3):
            new_image[i + 1, j + 1] = abs(np.sum(img[i : i + 3, j : j + 3] * L_sunnzi))
    return np.uint8(new_image)


"""
cv2.IMREAD_COLOR:读取一副彩色图片，图片的透明度会被忽略，默认为该值，实际取值为1；
cv2.IMREAD_GRAYSCALE:以灰度模式读取一张图片，实际取值为0
cv2.IMREAD_UNCHANGED:加载一副彩色图像，透明度不会被忽略。
"""
plt.rcParams["figure.figsize"] = (15, 15)
image = cv2.imread(r"picture\Lena.Bmp", cv2.IMREAD_GRAYSCALE)
plt.subplot(231)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Origin")
prewitt_img = prewitt(image)
plt.subplot(232)
plt.imshow(cv2.cvtColor(prewitt_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Prewitt")
robert_img = robert1(image)
plt.subplot(233)
plt.imshow(cv2.cvtColor(robert_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Robert")
sobel_img = sobel(image)
plt.subplot(234)
plt.imshow(cv2.cvtColor(sobel_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Sobel")
laplace_img = laplace(image)
plt.subplot(235)
plt.imshow(cv2.cvtColor(laplace_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Laplace")
plt.show()


plt.rcParams["figure.figsize"] = (15, 15)
prewitt_img_eq = cv2.equalizeHist(prewitt_img)
plt.subplot(221)
plt.imshow(cv2.cvtColor(prewitt_img_eq, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Prewitt_eq")
robert_img_eq = cv2.equalizeHist(robert_img)
plt.subplot(222)
plt.imshow(cv2.cvtColor(robert_img_eq, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Robert_eq")
sobel_img_eq = cv2.equalizeHist(sobel_img)
plt.subplot(223)
plt.imshow(cv2.cvtColor(sobel_img_eq, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Sobel_eq")
laplace_img_eq = cv2.equalizeHist(laplace_img)
plt.subplot(224)
plt.imshow(cv2.cvtColor(laplace_img_eq, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Laplace_eq")
plt.show()
