from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定义一个函数来计算图像的平均灰度值
def average_grayscale(image_path):
    image = Image.open(image_path).convert('L')# 打开图像并转换为灰度图
    width, height = image.size  # 获取图像宽度和高度
    pixels = list(image.getdata()) # 获取图像数据的像素值列表
    average_value = sum(pixels) / len(pixels)
    return average_value

# 定义一个函数来计算图像RGB通道的协方差矩阵
def Cov(image_path):
    image = cv2.imread(image_path)# 使用OpenCV读取图像
    w = image.shape[0]
    h = image.shape[1]# 获取宽度和高度
    b = [0] * w * h
    g = [0] * w * h
    r = [0] * w * h# 将图像分解为B, G, R三个通道，并转置以分别获取单独的通道数组

    for i in range(w):  # 获得BGR三个波段的像素值
        for j in range(h):
            for k in range(1):
                B = image[i, j, k + 0]
                G = image[i, j, k + 1]
                R = image[i, j, k + 2]
                b[i * h + j] = B
                g[i * h + j] = G
                r[i * h + j] = R
    # 计算平均值
    all_b = 0  # 像素值总和
    all_g = 0
    all_r = 0
    for i in range(w * h):
        all_b += b[i]
        all_g += g[i]
        all_r += r[i]
    mean_b = int(all_b / (w * h - 1))  # 像素值平均值 取整
    mean_g = int(all_g / (w * h - 1))
    mean_r = int(all_r / (w * h - 1))

    # 计算协方差
    cov_bb = 0
    cov_bg = 0
    cov_br = 0
    cov_gb = 0
    cov_gg = 0
    cov_gr = 0
    cov_rb = 0
    cov_rg = 0
    cov_rr = 0
    for i in range(w * h):
        cov_bb += (b[i] - mean_b) * (b[i] - mean_b)
        cov_bg += (b[i] - mean_b) * (g[i] - mean_g)
        cov_br += (b[i] - mean_b) * (r[i] - mean_r)
        cov_gb += (g[i] - mean_g) * (b[i] - mean_b)
        cov_gg += (g[i] - mean_g) * (g[i] - mean_g)
        cov_gr += (g[i] - mean_g) * (r[i] - mean_r)
        cov_rb += (r[i] - mean_r) * (b[i] - mean_b)
        cov_rg += (r[i] - mean_r) * (g[i] - mean_g)
        cov_rr += (r[i] - mean_r) * (r[i] - mean_r)
    m = w * h - 1
    print('协方差矩阵:')
    print(cov_bb / m, '\t', cov_bg / m, '\t', cov_br / m, '\n',
          cov_gb / m, '\t', cov_gg / m, '\t', cov_gr / m, '\n',
          cov_rb / m, '\t', cov_rg / m, '\t', cov_rr / m, '\n')


def pixel_operation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imshow('input', img)

    # 注意：python中的print函数默认换行，可以用end=''或者接任意字符
    # 像素均值、方差
    means, dev = cv2.meanStdDev(img)
    print('means: {}, \n dev: {}'.format(means, dev))
    # 像素最大值和最小值
    min_pixel = np.min(img[:, :, 0])
    max_pixel = np.max(img[:, :, -1])
    print('min: {}, max: {}'.format(min_pixel, max_pixel))

    # 若是一个空白图像
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    # 像素均值、方差
    # blank[:, :] = (255, 0, 255)
    means, dev = cv2.meanStdDev(blank)
    print('means: {}, \n dev: {}'.format(means, dev))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

file_path = 'picture/pepper.tif'
average_value = average_grayscale(file_path)
print(average_value)
pixel_operation(file_path)

Cov(file_path)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('picture/Lena.Bmp', 0)

img1 = img.astype('float')

C_temp = np.zeros(img.shape)
dst = np.zeros(img.shape)

m, n = img.shape
N = n
C_temp[0, :] = 1 * np.sqrt(1 / N)

for i in range(1, m):
    for j in range(n):
        C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)) * np.sqrt(2 / N)

dst = np.dot(C_temp, img1)
dst = np.dot(dst, np.transpose(C_temp))

dst1 = np.log(abs(dst))  # 进行log处理

img_recor = np.dot(np.transpose(C_temp), dst)
img_recor1 = np.dot(img_recor, C_temp)

# 自带方法

img_dct = cv2.dct(img1)  # 进行离散余弦变换

img_dct_log = np.log(abs(img_dct))  # 进行log处理

img_recor2 = cv2.idct(img_dct)  # 进行离散余弦反变换

plt.subplot(231)
plt.imshow(img1, 'gray')
plt.title('original image')
plt.xticks([]), plt.yticks([])

plt.subplot(232)
plt.imshow(dst1)
plt.title('DCT1')
plt.xticks([]), plt.yticks([])

plt.subplot(233)
plt.imshow(img_recor1, 'gray')
plt.title('IDCT1')
plt.xticks([]), plt.yticks([])

plt.subplot(234)
plt.imshow(img, 'gray')
plt.title('original image')

plt.subplot(235)
plt.imshow(img_dct_log)
plt.title('DCT2(cv2_dct)')

plt.subplot(236)
plt.imshow(img_recor2, 'gray')
plt.title('IDCT2(cv2_idct)')

plt.show()
