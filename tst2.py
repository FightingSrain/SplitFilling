
import cv2
import numpy as np
import matplotlib.pyplot as plt
# p = .125
# for i in range(100):
#     cont_cond = np.random.rand() < (1 - p)
#     print(cont_cond)
hint_line = np.zeros((1, 3, 512, 512), np.uint8)
# print(type(hint_line))
# hint_line = cv2.imread("./img_test/6.png") * 0
# print(type(hint_line))
# hint_line = np.expand_dims(hint_line, 0).transpose(0, 3, 1, 2) / 255.
print(hint_line.shape)
ptStart = (100, 100)
ptEnd = (200, 200)
point_color = (0,
                255,
                0)  # BGR 线的颜色
thickness = 3  # 线的宽度
lineType = 8
print(hint_line[0].transpose(1, 2, 0).shape)
tmp = hint_line[0].transpose(1, 2, 0)
cv2.line(np.transpose(hint_line[0], (1, 2, 0)), ptStart, ptEnd, point_color, thickness, lineType)
plt.imshow(hint_line[0].squeeze().transpose(1, 2, 0))
plt.title('Clustered1 Image')
plt.show()