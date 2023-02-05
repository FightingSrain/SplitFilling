
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random_walk import RandomWalk

# batch_mask batch_flat 都为[0, 1]
def scri_func(batch_mask, batch_flat, p=.125):
    B, _, H, W = batch_flat.shape
    hint_line = np.zeros((B, 3, H, W))  # BGRA
    hint_mask = np.zeros((B, 1, H, W))
    for nn in range(B):
        cont_cond = True
        ty = 0
        while (cont_cond):
            # 如果没有定义点个数，从几何分布随机采样点的个数
            cont_cond = np.random.rand() < (1 - p)  # p越小，采样的点越多
            ty += 1
            if ty == 5:
                cont_cond = False
            # print(cont_cond)
            # if (not cont_cond):  # skip out of loop if condition not met
            #     continue
            flat_img = batch_flat[nn, 0:3].transpose(1, 2, 0)
            mask = batch_mask[nn].transpose(1, 2, 0)
            _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)  # 二值化
            mask_img = (flat_img * mask.astype(np.uint8))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            mask_img = cv2.dilate(mask_img, kernel=kernel)
            # ----------
            H, W, C = mask_img.shape
            lstx = np.arange(0, H, 2)
            lsty = np.arange(0, W, 2)
            x1, y1 = -1, -1
            x2, y2 = -1, -1
            t = 0
            while t < 100: # 随机生成笔画上限
                x1 = lstx[np.random.randint(0, len(lstx))]
                y1 = lsty[np.random.randint(0, len(lsty))]
                t += 1
                if mask[x1, y1, 0] > 0 \
                        and mask[x1, y1, 1] > 0 \
                        and mask[x1, y1, 2] > 0:
                    # 随机游走3步
                    rw = RandomWalk()
                    x2, y2 = rw.fill_walk(x1, y1, mask, hint_line[nn].transpose(1, 2, 0))
                    break
            # 如果(x1，y1) (x2，y2) 合法
            if x1 != -1 and x2 != -1:
                ptStart = (y1, x1)
                ptEnd = (y2, x2)
                point_color = (float(mask_img[x1, y1, 0]),
                           float(mask_img[x1, y1, 1]),
                           float(mask_img[x1, y1, 2]))  # BGR 线的颜色
                thickness = 3  # 线的宽度
                lineType = 8 # 线条类型
                tmp_line = np.zeros((H, W, 3), np.float32)
                alpha = np.zeros((H, W), np.float32)
                cv2.line(tmp_line, ptStart, ptEnd, point_color, thickness, lineType)
                # tmp_line = cv2.cvtColor(tmp_line.astype(np.float32), cv2.COLOR_BGR2BGRA)  # BGR -> BGRA
                # print(tmp_line[:, :, 3])
                # print("---re----")
                # plt.imshow(tmp_line[:, :, 3])
                # plt.title('Clustered1 Image')
                # plt.show()
                # 赋值可以避免色彩累加
                hint_line[nn, 0:3, :, :] = np.where(hint_line[nn, 0:3, :, :] != 0, hint_line[nn, 0:3, :, :], tmp_line.transpose(2, 0, 1))
                # hint_line[nn, 3, :, :] = np.where(tmp_line.transpose(2, 0, 1)[0, :, :] != 0
                #                                   , 1., 0.)
                # plt.imshow(hint_line[nn, 3, :, :])
                # plt.title('Clustered1 Image')
                # plt.show()
                cv2.line(hint_mask[nn].squeeze(), ptStart, ptEnd, (1, 1), thickness, lineType)
    # hint_line = batch_flat * hint_mask
    return hint_line, hint_mask

# test
masks = cv2.imread("./img_res/res.jpg") / 255.
flat_img = cv2.imread("./img_test/6.png") / 255.  # 要4通道包括alpha通道
print(flat_img.shape)
print("====")
masks = np.expand_dims(masks, 0).transpose(0, 3, 1, 2)
flat_img = np.expand_dims(flat_img, 0).transpose(0, 3, 1, 2)
r1, r2 = scri_func(masks, flat_img)
print(r1.shape)
print(r2.shape)
cv2.imshow("res", r1.squeeze().transpose(1, 2, 0))
cv2.waitKey(0)
# plt.imshow(r1.squeeze().transpose(1, 2, 0))
# plt.title('Clustered1 Image')
# plt.show()

plt.imshow(r2.squeeze())
plt.title('Clustered1 Image')
plt.show()