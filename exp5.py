import cv2
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt

img = cv2.imread(r"picture/pepper.tif", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_arr = np.array(img)
img = img_arr.astype(np.double) / 255.0

F = fft.fft2(img, axes=(0, 1))  # 换到频域，信号在低频，噪声在高频
F = fft.fftshift(F, axes=(0, 1))  # 频谱移到矩阵中心

H, W = F.shape[:2]
u, v = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")  # 生成矩阵

# k=0.0025剧烈湍流；k=0.001中等湍流；k=0.00025低湍流
H_turbulence = np.exp(-0.0025 * ((u - H / 2) ** 2 + (v - W / 2) ** 2) ** (5 / 6))
H_turbulence = np.repeat(H_turbulence[:, :, np.newaxis], 3, axis=2)  # 对应RGB通道
F = F * H_turbulence

# 傅里叶反变换
X = fft.ifftshift(F, axes=(0, 1))
turimg = fft.ifft2(X, axes=(0, 1))
turimg = abs(turimg)
turimg = np.uint8(turimg * 255)


# def add_gaussian_noise(image, mean=0, sigma=25):
#     """
#     为图像添加高斯白噪声
#     :param image: 输入图像
#     :param mean: 高斯噪声的均值
#     :param sigma: 高斯噪声的标准差
#     :return: 添加噪声后的图像
#     """
#     # 生成高斯噪声
#     gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)

#     # 将噪声添加到图像
#     noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)

#     # 确保像素值在有效范围内
#     noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

#     return noisy_image


# def motion_process(image_size, motion_angle):
#     PSF = np.zeros(image_size)
#     print(image_size)
#     center_position = (image_size[0] - 1) / 2
#     print(center_position)

#     slope_tan = math.tan(motion_angle * math.pi / 180)
#     slope_cot = 1 / slope_tan
#     if slope_tan <= 1:
#         for i in range(15):
#             offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
#             PSF[int(center_position + offset), int(center_position - offset)] = 1
#         return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
#     else:
#         for i in range(15):
#             offset = round(i * slope_cot)
#             PSF[int(center_position - offset), int(center_position + offset)] = 1
#         return PSF / PSF.sum()


# def make_blurred(input, PSF, eps):
#     input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
#     PSF_fft = fft.fft2(PSF) + eps
#     blurred = fft.ifft2(input_fft * PSF_fft)
#     blurred = np.abs(fft.fftshift(blurred))
#     return blurred


# def inverse(input, PSF, eps):  # 逆滤波
#     input_fft = fft.fft2(input)
#     PSF_fft = fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
#     result = fft.ifft2(input_fft / PSF_fft)  # 计算F(u,v)的傅里叶反变换
#     result = np.abs(fft.fftshift(result))
#     return result


# def wiener(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
#     input_fft = fft.fft2(input)
#     PSF_fft = fft.fft2(PSF) + eps
#     PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
#     result = fft.ifft2(input_fft * PSF_fft_1)
#     result = np.abs(fft.fftshift(result))
#     return result


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


kernel = np.ones((H, W), np.float32) / (H * W)

gasuss_img = gaussian_noise(turimg)
# gasuss_img1 = cv2.GaussianBlur(gasuss_img, (3, 3), 0)
inverse_img = inverse_filtering(gasuss_img.astype(np.uint8), kernel)

noise_var = 25**2  # Assuming noise standard deviation is 25
wiener_img = wiener_filtering(gasuss_img.astype(np.uint8), kernel, noise_var)
# wiwner_img = cv2.Wiener2(turimg, (5, 5), (10, 10))

# PSF = motion_process((W, H), 60)
# blurred = np.abs(make_blurred(turimg, PSF, 1e-3))

# inverse_img = inverse(turimg, PSF, 1e-3)
# wiener_img = wiener(turimg, PSF, 1e-3)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["figure.figsize"] = (15, 15)
plt.subplot(231)
plt.imshow(img)
plt.title("原图像")
plt.axis("off")
plt.subplot(232)
plt.imshow(turimg)
plt.title("退化图像")
plt.axis("off")
plt.subplot(233)
plt.imshow(gasuss_img)
plt.title("噪声图像")
plt.axis("off")
plt.subplot(234)
plt.imshow(inverse_img)
plt.title("逆滤波图像")
plt.axis("off")
plt.subplot(235)
plt.imshow(wiener_img)
plt.title("维纳滤波图像")
plt.axis("off")
plt.show()
