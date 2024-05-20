import cv2
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt


# 9.19: 湍流模糊退化模型 (turbulence blur degradation model)
def getDegradedImg(image, Huv):  # 根据退化模型生成退化图像
    rows, cols = image.shape[:2]  # 图片的高度和宽度
    # (1) 中心化, centralized 2d array f(x,y) * (-1)^(x+y)
    mask = np.ones((rows, cols))
    mask[1::2, ::2] = -1
    mask[::2, 1::2] = -1
    imageCen = image * mask
    # (2) 快速傅里叶变换
    dftImage = np.zeros((rows, cols, 2), np.float32)
    dftImage[:, :, 0] = imageCen
    cv2.dft(
        dftImage, dftImage, cv2.DFT_COMPLEX_OUTPUT
    )  # 快速傅里叶变换 (rows, cols, 2)
    # (4) 构建 频域滤波器传递函数:
    Filter = np.zeros((rows, cols, 2), np.float32)  # (rows, cols, 2)
    Filter[:, :, 0], Filter[:, :, 1] = Huv, Huv
    # (5) 在频率域修改傅里叶变换: 傅里叶变换 点乘 滤波器传递函数
    dftFilter = dftImage * Filter
    # (6) 对修正傅里叶变换 进行傅里叶逆变换，并只取实部
    idft = np.ones((rows, cols), np.float32)  # 快速傅里叶变换的尺寸
    cv2.dft(
        dftFilter, idft, cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE + cv2.DFT_SCALE
    )  # 只取实部
    # (7) 中心化, centralized 2d array g(x,y) * (-1)^(x+y)
    mask2 = np.ones(dftImage.shape[:2])
    mask2[1::2, ::2] = -1
    mask2[::2, 1::2] = -1
    idftCen = idft * mask2  # g(x,y) * (-1)^(x+y)
    # (8) 截取左上角，大小和输入图像相等
    imgDegraded = np.uint8(
        cv2.normalize(idftCen, None, 0, 255, cv2.NORM_MINMAX)
    )  # 归一化为 [0,255]
    # print(image.shape, dftFilter.shape, imgDegraded.shape)
    return imgDegraded


def turbulenceBlur(img, k=0.001):  # 湍流模糊传递函数
    # H(u,v) = exp(-k(u^2+v^2)^5/6)
    M, N = img.shape[1], img.shape[0]
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    radius = (u - M // 2) ** 2 + (v - N // 2) ** 2
    kernel = np.exp(-k * np.power(radius, 5 / 6))
    return kernel


def gaussian_noise(image, mean=0, std_dev=25):
    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, image.shape)

    # Add noise to image
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return noisy_image


def inverse_filtering(image, kernel):
    # Perform inverse filtering
    restored_image = cv2.filter2D(image, -1, np.linalg.pinv(kernel))

    return restored_image


def wiener_filtering(image, kernel, noise_var):
    # Calculate Wiener filter
    h_fft = fft.fft2(kernel)
    h_fft_conj = np.conj(h_fft)
    h_fft_sq = np.abs(h_fft) ** 2
    snr = 1 / noise_var
    wiener_filter = h_fft_conj / (h_fft_sq + snr)

    # Apply Wiener filter
    restored_image = fft.ifft2(fft.fft2(image) * wiener_filter).real

    return np.clip(restored_image, 0, 255).astype(np.uint8)


img = cv2.imread(r"picture/pepper.tif", cv2.IMREAD_GRAYSCALE)

# 生成湍流模糊图像
HBlur = turbulenceBlur(img, k=0.001)  # 湍流模糊传递函数
imgBlur = getDegradedImg(img, HBlur)  # 生成湍流模糊图像

gaussian_img1 = gaussian_noise(imgBlur)

inverse_img = inverse_filtering(gaussian_img1, HBlur)
wiener_img = wiener_filtering(gaussian_img1, HBlur, imgBlur)

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
plt.imshow(gaussian_img1, "gray")
plt.title("高斯噪声图像")
plt.axis("off")
plt.subplot(236)
plt.imshow(inverse_img, "gray")
plt.title("逆滤波图像")
plt.axis("off")
plt.show()
