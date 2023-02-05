
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# 读取mask图像
masks = cv2.imread("./img_res/res.jpg")/255
# masks = cv2.medianBlur(masks, 5)  # 去除椒盐噪声
_, mask = cv2.threshold(masks, 0.5, 1, cv2.THRESH_BINARY)
# mask = mask // 255  # 归一化

# 读取falt图像
flat_img = cv2.imread("./img_test/1.jpg")/255
# 得到mask flat 图像
mask_img = (flat_img * mask.astype(np.uint8))
# 膨胀15个像素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
mask_img = cv2.dilate(mask_img, kernel=kernel)

# 生成模拟线的 起始 终止 点
H, W, C = mask_img.shape
lstx = np.arange(0, H, 2)
lsty = np.arange(0, W, 2)
while True:
    x1 = lstx[np.random.randint(0, len(lstx))]
    y1 = lsty[np.random.randint(0, len(lsty))]
    if mask[x1, y1, 0] > 0 \
            and mask[x1, y1, 1] > 0 \
            and mask[x1, y1, 2] > 0:
        break
while True:
    x2 = lstx[np.random.randint(0, len(lstx))]
    y2 = lsty[np.random.randint(0, len(lsty))]
    if mask[x2, y2, 0] > 0 \
            and mask[x2, y2, 1] > 0 \
            and mask[x2, y2, 2] > 0 \
            and np.sqrt((x2 - x1)**2 + (y2 - y1)**2) < min(H//6, W//6):
        break
# x2 = np.clip(np.random.randint(x1-H//6, x1+H//6), a_min=0, a_max=H-1)
# y2 = np.clip(np.random.randint(y1-W//6, y1+W//6), a_min=0, a_max=W-1)

hint_line = np.zeros_like(flat_img)
hint_mask = np.zeros((H, W))

ptStart = (y1, x1)
ptEnd = (y2, x2)
point_color = ((mask_img[x1, y1, 0]),
               (mask_img[x1, y1, 1]),
               (mask_img[x1, y1, 2]))  # BGR 线的颜色
thickness = 3 # 线的宽度
lineType = 8
cv2.line(mask_img, ptStart, ptEnd, (255, 0, 0), thickness, lineType)
cv2.line(hint_line, ptStart, ptEnd, point_color, thickness, lineType)
cv2.line(hint_mask, ptStart, ptEnd, (1, 1, 1), thickness, lineType)
plt.imshow((hint_line))
plt.title('Clustered1 Image')
plt.show()
plt.imshow(hint_mask)
plt.title('Clustered2 Image')
plt.show()
plt.imshow(mask_img)
plt.title('Clustered3 Image')
plt.show()







