import cv2
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt


def getDegradedImg(image, Huv):  # 根据退化模型生成退化图像
    # (1) 傅里叶变换, 中心化
    F = fft.fft2(image.astype(np.float32))  # 傅里叶变换
    fftShift = fft.fftshift(F)  # 将低频分量移动到频域图像中心
    # (2) 在频率域修改傅里叶变换: 傅里叶变换 点乘 滤波器传递函数
    fftShiftFilter = fftShift * Huv  # Guv = Fuv * Huv
    # (3) 对修正傅里叶变换 进行傅里叶逆变换，逆中心化
    invShift = fft.ifftshift(fftShiftFilter)  # 将低频分量逆转换回图像四角
    imgIfft = fft.ifft2(invShift)  # 逆傅里叶变换，返回值是复数数组
    imgDegraded = np.uint8(
        cv2.normalize(np.abs(imgIfft), None, 0, 255, cv2.NORM_MINMAX)
    )  # 归一化为 [0,255]
    return imgDegraded


def turbulenceBlur(img, k=0.001):  # 湍流模糊传递函数
    # H(u,v) = exp(-k(u^2+v^2)^5/6)
    M, N = img.shape[1], img.shape[0]
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    radius = (u - M // 2) ** 2 + (v - N // 2) ** 2
    kernel = np.exp(-k * np.power(radius, 5 / 6))
    return kernel


def gaussian_noise(image, mean=0, sigma=20.0):
    noiseGause = np.random.normal(mean, sigma, image.shape)
    imgGaussNoise = image + noiseGause
    imgGaussNoise = np.uint8(
        cv2.normalize(imgGaussNoise, None, 0, 255, cv2.NORM_MINMAX)
    )  # 归一化为 [0,255]
    return imgGaussNoise


def ideaLPFilter(img, radius=10):  # 理想低通滤波器
    M, N = img.shape[1], img.shape[0]
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    D = np.sqrt((u - M // 2) ** 2 + (v - N // 2) ** 2)
    kernel = np.zeros(img.shape[:2], np.float32)
    kernel[D <= radius] = 1
    return kernel


def inverseFilter(image, Huv, D0):  # 根据退化模型逆滤波
    # (1) 傅里叶变换, 中心化
    F = fft.fft2(image.astype(np.float32))  # 傅里叶变换
    fftShift = fft.fftshift(F)  # 将低频分量移动到频域图像中心
    # (2) 在频率域修改傅里叶变换: 傅里叶变换 点乘 滤波器传递函数
    if D0 == 0:
        fftShiftFilter = fftShift / Huv  # Guv = Fuv / Huv
    else:
        lpFilter = ideaLPFilter(image, radius=D0)
        fftShiftFilter = fftShift / Huv * lpFilter  # Guv = Fuv / Huv
    # (3) 对修正傅里叶变换 进行傅里叶逆变换，逆中心化
    invShift = fft.ifftshift(fftShiftFilter)  # 将低频分量逆转换回图像四角
    imgIfft = fft.ifft2(invShift)  # 逆傅里叶变换，返回值是复数数组
    imgRebuild = np.uint8(
        cv2.normalize(np.abs(imgIfft), None, 0, 255, cv2.NORM_MINMAX)
    )  # 归一化为 [0,255]
    return imgRebuild


def wienerFilter(image, Huv, eps, K=0.01):  # 维纳滤波，K=0.01
    F = fft.fft2(image.astype(np.float32))
    fftShift = fft.fft2(Huv) + eps
    fftWiener = np.conj(fftShift) / (np.abs(fftShift) ** 2 + K)
    imgWienerFilter = fft.ifft2(F * fftWiener)
    imgWienerFilter = np.abs(fft.fftshift(imgWienerFilter))
    return imgWienerFilter


img = cv2.imread(r"picture/pepper.tif", cv2.IMREAD_GRAYSCALE)

# 生成湍流模糊图像
HBlur = turbulenceBlur(img)
imgBlur = np.abs(getDegradedImg(img, HBlur))

gaussian_img = gaussian_noise(imgBlur)

inverse_img = inverseFilter(gaussian_img, HBlur, D0=100)
wiener_img = wienerFilter(gaussian_img, HBlur, 1e-6)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["figure.figsize"] = (15, 15)
plt.subplot(231)
plt.imshow(img, "gray")
plt.title("原图像")
plt.axis("off")
plt.subplot(232)
plt.imshow(imgBlur, "gray")
plt.title("退化图像")
plt.axis("off")
plt.subplot(233)
plt.imshow(gaussian_img, "gray")
plt.title("高斯噪声图像")
plt.axis("off")
plt.subplot(234)
plt.imshow(inverse_img, "gray")
plt.title("逆滤波图像")
plt.axis("off")
plt.subplot(235)
plt.imshow(wiener_img, "gray")
plt.title("维纳滤波图像")
plt.axis("off")
plt.show()
