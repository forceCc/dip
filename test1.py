import cv2
from matplotlib import pyplot as plt
import matplotlib

img = cv2.imread("picture/pepper.tif")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray_img)
cv2.waitKey(0)

b, g, r = cv2.split(img)
image = cv2.merge((r, g, b))

plt.rcParams["figure.figsize"] = (15, 15)
plt.subplot(121)
plt.imshow(image, cmap=plt.cm.gray)
plt.axis("off")
plt.title("src")
plt.subplot(122)
plt.imshow(gray_img, cmap=plt.cm.gray)
plt.axis("off")
plt.title("gray")
plt.show()

a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
for i in a:
    print(i)
