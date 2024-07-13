import cv2
import numpy as np
from matplotlib import pyplot as plt


# 9.21: 退化图像的维纳滤波 (Wiener filter)
def getMotionDsf(shape, angle, dist):
    xCenter = (shape[0] - 1) / 2
    yCenter = (shape[1] - 1) / 2
    sinVal = np.sin(angle * np.pi / 180)
    cosVal = np.cos(angle * np.pi / 180)
    PSF = np.zeros(shape)  # 点扩散函数
    for i in range(dist):  # 将对应角度上motion_dis个点置成1
        xOffset = round(sinVal * i)
        yOffset = round(cosVal * i)
        PSF[int(xCenter - xOffset), int(yCenter + yOffset)] = 1
    return PSF / PSF.sum()  # 归一化


def makeBlurred(image, PSF, eps):  # 对图片进行运动模糊
    fftImg = np.fft.fft2(image)  # 进行二维数组的傅里叶变换
    fftPSF = np.fft.fft2(PSF) + eps
    fftBlur = np.fft.ifft2(fftImg * fftPSF)
    fftBlur = np.abs(np.fft.fftshift(fftBlur))
    return fftBlur


def inverseFilter(image, PSF, eps):  # 逆滤波
    fftImg = np.fft.fft2(image)
    fftPSF = np.fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
    imgInvFilter = np.fft.ifft2(fftImg / fftPSF)  # 计算F(u,v)的傅里叶反变换
    imgInvFilter = np.abs(np.fft.fftshift(imgInvFilter))
    return imgInvFilter


def wienerFilter(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    fftImg = np.fft.fft2(input)
    fftPSF = np.fft.fft2(PSF) + eps
    fftWiener = np.conj(fftPSF) / (np.abs(fftPSF) ** 2 + K)
    imgWienerFilter = np.fft.ifft2(fftImg * fftWiener)
    imgWienerFilter = np.abs(np.fft.fftshift(imgWienerFilter))
    return imgWienerFilter


# 读取原始图像
img = cv2.imread("picture\pepper.tif", 0)  # flags=0 读取为灰度图像
hImg, wImg = img.shape[:2]

# 不含噪声的运动模糊
PSF = getMotionDsf((hImg, wImg), 45, 100)  # 运动模糊函数
imgBlurred = np.abs(makeBlurred(img, PSF, 1e-6))  # 生成不含噪声的运动模糊图像
imgInvFilter = inverseFilter(imgBlurred, PSF, 1e-6)  # 逆滤波
imgWienerFilter = wienerFilter(imgBlurred, PSF, 1e-6)  # 维纳滤波

# 带有噪声的运动模糊
scale = 0.05  # 噪声方差
noisy = imgBlurred.std() * np.random.normal(
    loc=0.0, scale=scale, size=imgBlurred.shape
)  # 添加高斯噪声
imgBlurNoisy = imgBlurred + noisy  # 带有噪声的运动模糊
imgNoisyInv = inverseFilter(imgBlurNoisy, PSF, scale)  # 对添加噪声的模糊图像进行逆滤波
imgNoisyWiener = wienerFilter(
    imgBlurNoisy, PSF, scale
)  # 对添加噪声的模糊图像进行维纳滤波

plt.figure(figsize=(9, 7))
(
    plt.subplot(231),
    plt.title("blurred image"),
    plt.axis("off"),
    plt.imshow(imgBlurred, "gray"),
)
(
    plt.subplot(232),
    plt.title("inverse filter"),
    plt.axis("off"),
    plt.imshow(imgInvFilter, "gray"),
)
(
    plt.subplot(233),
    plt.title("Wiener filter"),
    plt.axis("off"),
    plt.imshow(imgWienerFilter, "gray"),
)
(
    plt.subplot(234),
    plt.title("blurred image with noisy"),
    plt.axis("off"),
    plt.imshow(imgBlurNoisy, "gray"),
)
(
    plt.subplot(235),
    plt.title("inverse filter"),
    plt.axis("off"),
    plt.imshow(imgNoisyInv, "gray"),
)
(
    plt.subplot(236),
    plt.title("Wiener filter"),
    plt.axis("off"),
    plt.imshow(imgNoisyWiener, "gray"),
)
plt.tight_layout()
plt.show()
