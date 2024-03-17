import cv2
import numpy as np

# 读取两个图像
img1 = cv2.imread(r'C:\Users\SiChengLi\Desktop\5-fixed.png')
img2 = cv2.imread(r'C:\Users\SiChengLi\Desktop\5-undistorted.png')

# 计算PSNR
mse = np.mean((img1 - img2) ** 2)
if mse == 0:
    psnr = 100
else:
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

print("PSNR: ", psnr)

# 计算信息熵
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

hist1, bins1 = np.histogram(gray_img1.ravel(), 256, [0, 256])
hist2, bins2 = np.histogram(gray_img2.ravel(), 256, [0, 256])

entropy1 = -np.sum(np.multiply(hist1, np.log2(hist1 + 1e-7)))
entropy2 = -np.sum(np.multiply(hist2, np.log2(hist2 + 1e-7)))

print("Entropy of image 1: ", entropy1)
print("Entropy of image 2: ", entropy2)