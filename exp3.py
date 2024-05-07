import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread(r"picture/Lena.Bmp")
# 将图像转为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对图像进行平滑操作
smooth = cv2.blur(gray_img, (9, 9))
box = cv2.boxFilter(gray_img, -1, (3, 3), normalize=0)
median = cv2.medianBlur(gray_img, 9)
Gausi = cv2.GaussianBlur(gray_img, (9, 9), 1)  # 高斯滤波

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.subplot(231)
plt.imshow(gray_img, cmap=plt.cm.gray)
plt.axis("off")
plt.title("原始图像")
plt.subplot(232)
plt.imshow(smooth, cmap=plt.cm.gray)
plt.axis("off")
plt.title("均值滤波")
plt.subplot(233)
plt.imshow(box, cmap=plt.cm.gray)
plt.axis("off")
plt.title("BOX滤波")
plt.subplot(234)
plt.imshow(median, cmap=plt.cm.gray)
plt.axis("off")
plt.title("中值滤波")
plt.subplot(235)
plt.imshow(Gausi, cmap=plt.cm.gray)
plt.axis("off")
plt.title("高斯滤波")
plt.show()

source = cv2.imread("picture/1z/xust_yuantu.bmp")
chengxing = cv2.imread("picture/1z/xust_chengxing.bmp")
gaosi = cv2.imread("picture/1z/xust_gaosi.bmp")
jiaoyan = cv2.imread("picture/1z/xust_jiaoyan.bmp")
source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
chengxing = cv2.cvtColor(chengxing, cv2.COLOR_BGR2GRAY)
gaosi = cv2.cvtColor(gaosi, cv2.COLOR_BGR2GRAY)
jiaoyan = cv2.cvtColor(jiaoyan, cv2.COLOR_BGR2GRAY)


def medfilter(pic, scale=[5, 5], pad=[0, 0, 0], type="cross"):
    """
    中值滤波器
    pic 为被作用图片
    type  为中值窗口类型 可选参数为 'cross'
    scale 为窗口大小，如[3,3]
    pad   为填充方法 输入值为一维矩阵
    """

    # 分别获取用于常数填充的四周填充的距离
    top_bottom = int((scale[1] - 1) / 2)
    left_right = int((scale[0] - 1) / 2)
    # 获取用于中值滤波窗口的中值位数（如3×3中，5个数排序后取第三个数）
    mednum = (scale[1] + scale[0]) / 2 - 1
    total_dim = np.shape(pic)
    # 获取图像的行列数据
    pic_line = total_dim[0]
    pic_row = total_dim[1]  # 列
    # 进行指定的边界填充
    pic = cv2.copyMakeBorder(
        pic,
        top_bottom,
        top_bottom,
        left_right,
        left_right,
        cv2.BORDER_CONSTANT,
        value=pad,
    )

    # 定义十字中滤波函数
    def crossfilt(pic=pic, pic_line=pic_line, pic_row=pic_row):
        for i in range(pic_line):
            for j in range(pic_row):  # 两个for循环遍历图像所有像素
                mask = [pic[i, j]]  # 确定十字形窗口中心
                for n in range(
                    1, top_bottom + 1
                ):  # 以第（i,j）个像素为中心点像四周十字形发散
                    arra = [pic[i, j + n], pic[i, j - n], pic[i - n, j], pic[i + n, j]]
                    mask = np.hstack((mask, arra))
                pic[i, j] = np.sort(mask)[int(mednum)]

    # 通过输入图像维度判断Gray或者RGB
    if len(total_dim) > 2:  # RGB图像
        r, g, b = cv2.split(pic)
        r, g, b = medfilter(r), medfilter(g), medfilter(b)
        pic = cv2.merge([r, g, b])
    elif len(total_dim) <= 2:  # 灰度图像
        crossfilt(pic)
    return pic


chengxing1 = medfilter(chengxing)
gaosi1 = medfilter(gaosi)
jiaoyan1 = medfilter(jiaoyan)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.subplot(221)
plt.imshow(source, cmap=plt.cm.gray)
plt.axis("off")
plt.title("原始图像")
plt.subplot(222)
plt.imshow(chengxing1, cmap=plt.cm.gray)
plt.axis("off")
plt.title("瑞利")
plt.subplot(223)
plt.imshow(gaosi1, cmap=plt.cm.gray)
plt.axis("off")
plt.title("高斯")
plt.subplot(224)
plt.imshow(jiaoyan1, cmap=plt.cm.gray)
plt.axis("off")
plt.title("椒盐")
plt.show()
