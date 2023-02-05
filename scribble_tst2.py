
import numpy as np

import cv2
import matplotlib.pyplot as plt
masks = cv2.imread("./img_res/res.jpg")
# masks = cv2.medianBlur(masks, 5)  # 去除椒盐噪声
_, mask = cv2.threshold(masks, 127, 255, cv2.THRESH_BINARY)
# 膨胀15个像素
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
# mask = cv2.dilate(mask, kernel=kernel)
mask = mask // 255  # 归一化

# mask = np.clip(mask.astype(np.uint8), a_max=0, a_min=1)
# print(mask.min())
# print(mask.max())
# print(mask.shape)
# plt.imshow(mask*255)
# plt.title('Clustered2 Image')
# plt.show()
# print(mask[529, 705]*255)
# print(")))))))))")
# print(mask.max())
# print(mask.min())
# print("======")
cv2.imshow("res1", mask)  # ((image_matrixs//255)*images).astype(np.int8)
cv2.waitKey(0)
# mask = np.where(mask > 0., 1, mask)
img = cv2.imread("./img_test/1.jpg")
# cv2.imshow("res2", (mask * 255).astype(np.uint8))  # ((image_matrixs//255)*images).astype(np.int8)
# cv2.waitKey(0)
cv2.imshow("res3", (img * mask).astype(np.uint8))  # ((image_matrixs//255)*images).astype(np.int8)
cv2.waitKey(0)

# plt.imshow(img * mask)
# plt.title('Clustered2 Image')
# plt.show()
# ===========================
mask_img = (img * mask.astype(np.uint8)).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
mask_img = cv2.dilate(mask_img, kernel=kernel)
H, W, C = mask_img.shape
print(mask_img.shape)
hint_line = np.zeros_like(img)
hint_mask = np.zeros((H, W))

# P = 3
x1 = -1
y1 = -1
x2 = -1
y2 = -1
# 生成区域内的随机点
lstx = np.arange(0, H, 5)
lsty = np.arange(0, W, 5)
# y = np.random.randint(0, len(lsty))
while True:
    x1 = lstx[np.random.randint(0, len(lstx))]
    y1 = lsty[np.random.randint(0, len(lsty))]
    if mask[x1, y1, 0] > 0 \
            and mask[x1, y1, 1] > 0 \
            and mask[x1, y1, 2] > 0:
        print(x1, y1)
        print(mask[x1][y1])
        print(mask_img[x1][y1])
        print("=====+=====")
        break
# =================
x2 = np.clip(np.random.randint(x1-100, x1+100), a_min=0, a_max=H-1)
y2 = np.clip(np.random.randint(y1-100, y1+100), a_min=0, a_max=W-1)
# while True:
#     x2 = np.random.randint(x1-7, x1+8)
#     y2 = np.random.randint(y1-7, y2+8)
    # if mask_img[x2, y2, 0] != 0 \
    #         and mask_img[x2, y2, 1] != 0 \
    #         and mask_img[x2, y2, 2] != 0:
    #     print(mask_img[x2, y2, 0])
    #     break
# print(mask_img)
# print(200, 200)
# print(600, 600)
print(mask_img[200, 200, 0])
print(mask_img[155, 484])
print(mask[155, 484])
print("==========")
ptStart = (y1, x1)
ptEnd = (y2, x2)
# point_color = (int(mask_img[x1][y1][0]),
#                int(mask_img[x1][y1][1]),
#                int(mask_img[x1][y1][2]))  # BGR
point_color = (0, 0, 255)  # BGR
print(point_color)
thickness = 3
lineType = 8
cv2.line(mask_img, ptStart, ptEnd, point_color, thickness, lineType)
cv2.line(hint_line, ptStart, ptEnd, point_color, thickness, lineType)
cv2.imwrite("./img_res/line.jpg", hint_line)
# print(mask)

# h = int(np.clip(np.random.normal((H - P + 1) / 2., (H - P + 1) / 4.), 0, H - P))
# w = int(np.clip(np.random.normal((W - P + 1) / 2., (W - P + 1) / 4.), 0, W - P))

cv2.imshow("res4", mask_img)  # ((image_matrixs//255)*images).astype(np.int8)
cv2.waitKey(0)
plt.imshow(mask_img)
plt.title('Clustered2 Image')
plt.show()



















