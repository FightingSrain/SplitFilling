

import numpy as np
# lst1 = np.arange(0, 1024, 5)
# a = np.random.randint(0, len(lst1))
# print(lst1[a])
# # print(lst1)




import numpy as np

import cv2
import matplotlib.pyplot as plt


masks = cv2.imread("./img_res/res.jpg", cv2.IMREAD_COLOR)
print(masks.shape)
# plt.imshow(masks[500-200: 520+200, 690-200: 710+200, :])
# plt.imshow(masks[690-100: 710+100, 500-100: 520+100, :])
# plt.title('Clustered3 Image')
# plt.show()
cv2.imshow("res", masks)  # ((image_matrixs//255)*images).astype(np.int8)
cv2.waitKey(0)
# masks = cv2.medianBlur(masks, 5)  # [0.0, 1.0] 去除椒盐噪声
_, mask = cv2.threshold(masks, 127, 255, cv2.THRESH_BINARY)
# mask = (mask // 255).astype('uint8')
print(mask.shape)
print("------dwf")
# 图像腐蚀
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
dst = cv2.erode(mask, kernel=kernel)
print(dst[510, 700])
print(">>>>>>>>")

# 像素膨胀 膨胀15个像素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
mask = cv2.dilate(mask, kernel=kernel)

cv2.circle(mask, (100, 700), 5, (255, 0, 0), 1, 8, 0)
cv2.circle(mask, (510, 700), 5, (255, 0, 0), 1, 8, 0)
plt.imshow(mask)
plt.title('Clustered2 Image')
plt.show()
print(mask[510][700])
print(dst[510, 700])
print(mask[510, 700])
print(mask[460, 200])
print("=========")
# print(mask[515, 600]*255)
print(")))))))))")
plt.imshow(dst)
plt.title('Clustered2 Image')
plt.show()


# ===============

# a = np.zeros((5, 5, 3), np.uint8)
# _, a = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
# print(a[2, 1])
# plt.imshow(a)
# plt.title('Clustered2 Image')
# plt.show()



