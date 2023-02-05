




import torch
import os
import cv2
import numpy as np
import torch.nn as nn
from mini_batch_loader import MiniBatchLoader
from scribble_func import scri_func
from Unet.unet import Net
from utils import init_net
import matplotlib.pyplot as plt

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(1234)

FLAT_PATH = "./path_file/flat.txt"
MASK0_PATH = "./path_file/mask0.txt"
MASK1_PATH = "./path_file/mask1.txt"
MASK2_PATH = "./path_file/mask2.txt"
MASK3_PATH = "./path_file/mask3.txt"
MASK4_PATH = "./path_file/mask4.txt"
MASK5_PATH = "./path_file/mask5.txt"
MASK6_PATH = "./path_file/mask6.txt"
MASK7_PATH = "./path_file/mask7.txt"

SKELETON_PATH = "./path_file/skeleton.txt"
LINE_PATH = "./path_file/line.txt"
IMAGE_DIR_PATH = ".//"

batch_size = 16  # 32
lr = 0.0001

def process():
    mini_batch_loader1 = MiniBatchLoader(
        FLAT_PATH,  # flat
        MASK0_PATH,  # mask0
        MASK1_PATH,  # mask1
        MASK2_PATH,  # mask2
        MASK3_PATH,  # mask3
        MASK4_PATH,  # mask4
        MASK5_PATH,  # mask5
        MASK6_PATH,  # mask6
        MASK7_PATH,  # mask7
        SKELETON_PATH,  # skeleton
        LINE_PATH,  # line
        IMAGE_DIR_PATH,
        256)

    train_data_size1 = MiniBatchLoader.count_paths(FLAT_PATH)
    indices1 = np.random.permutation(train_data_size1)

    model = Net().to(device)

    model = init_net(model, 'kaiming', gpu_ids=[]) # 正交初始化收敛慢，Kaiming初始化可以较好收敛

    mse = nn.MSELoss()
    opetimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # opetimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # model.load_state_dict(torch.load("./model_test1/modela5400_.pth"))
    model.load_state_dict(torch.load("./model_test1/modela2800_.pth"))

    i_index1 = 0
    for epi in range(100000):
        r = indices1[i_index1: i_index1 + batch_size]
        flat, mask0, mask1, mask2, mask3, mask4, \
        mask5, mask6, mask7, \
        skeleton, line = mini_batch_loader1.load_training_data(r)  #

        maskall = [mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7]
        hintline = []  # 提示线
        hintmask = []  # 提示线mask
        for i in range(len(maskall)):

            hint_line, hint_mask = scri_func(maskall[i], flat)
            hintline.append(hint_line)
            hintmask.append(hint_mask)
        hintsum = hintmask[0] + hintmask[1] + hintmask[2] + hintmask[3] + hintmask[4] + \
                  hintmask[5] + hintmask[6] + hintmask[7]  # 提示线mask的和

        # for i in range(len(hintline)): # 按顺序便利
        for _ in range(1):
            i = np.random.randint(0, 8)
            # hint line mask [-1, 0, 1]
            value = 1  # [-1, 0, 1]
            # cur = np.ones_like(hintsum) * value  # 32, 3, 256, 256
            tmp = np.clip(hintsum - hintmask[i], a_min=0, a_max=1)
            cur = np.where(tmp == 1, -value, 0)
            cur_mask = cur * (1 - hintmask[i]) + hintmask[i] * value  # 当前提示线mask
            # cur_mask += 0.5
            # 输入 line(线稿), hintline[i](提示线), cur_mask(提示线mask)
            # label flat, maskall[i], skeleton
            # print(line.shape)  # (32, 3, 256, 256)
            # print(hintline[i].shape)  # (32, 3, 256, 256)
            # print(cur_mask.shape)  # (32, 3, 256, 256)
            # # ----------
            # print(flat.shape)  # (32, 3, 256, 256)
            # print(maskall[i].shape)  # (32, 3, 256, 256)
            # print(skeleton.shape)  # (32, 3, 256, 256)
            print("==============")
            # plt.figure()
            # plt.subplot(2, 3, 1)
            # plt.imshow(line[0].transpose(1, 2, 0))
            #
            # plt.subplot(2, 3, 2)
            # plt.imshow(hintline[i][0].transpose(1, 2, 0))
            #
            # plt.subplot(2, 3, 3)
            # plt.imshow(cur_mask[0].transpose(1, 2, 0))
            #
            # plt.subplot(2, 3, 4)
            # plt.imshow(flat[0].transpose(1, 2, 0))
            #
            # plt.subplot(2, 3, 5)
            # plt.imshow(maskall[i][0].transpose(1, 2, 0))
            #
            # plt.subplot(2, 3, 6)
            # plt.imshow(skeleton[0].transpose(1, 2, 0))
            # plt.show()

            ins = np.concatenate([line[:, 0:1, :, :], hintline[i], cur_mask[:, 0:1, :, :]], 1)
            ins = torch.FloatTensor(ins).cuda()

            res_flat, res_inf, res_ske = model(ins)

            flats = torch.FloatTensor(flat).cuda()
            inf = torch.FloatTensor(maskall[i][:, 0:1, :, :]).cuda()
            ske = torch.FloatTensor(skeleton[:, 0:1, :, :]).cuda()

            loss1 = mse(res_flat, flats)
            loss2 = mse(res_inf, inf)
            loss3 = mse(res_ske, ske)
            # loss = loss1/loss1.detach() + loss2/loss2.detach() + loss3/loss3.detach()
            # ========
            loss = mse(res_flat, flats) + mse(res_inf, inf) + mse(res_ske, ske)

            # =============
            if epi % 10 == 0:
                plt.ion()
                plt.figure()
                plt.subplot(3, 3, 1)
                plt.imshow(res_flat[0].detach().cpu().numpy().transpose(1, 2, 0))

                plt.subplot(3, 3, 2)
                plt.imshow(res_inf[0].detach().cpu().numpy().squeeze())

                plt.subplot(3, 3, 3)
                plt.imshow(res_ske[0].detach().cpu().numpy().squeeze())

                plt.subplot(3, 3, 4)
                plt.imshow(flat[0].transpose(1, 2, 0))

                plt.subplot(3, 3, 5)
                plt.imshow(maskall[i][0].transpose(1, 2, 0))

                plt.subplot(3, 3, 6)
                plt.imshow(skeleton[0].transpose(1, 2, 0))

                plt.subplot(3, 3, 7)
                plt.imshow(line[0, 0, :, :])

                plt.subplot(3, 3, 8)
                plt.imshow(hintline[i][0, 0:3, :, :].transpose(1, 2, 0))

                plt.subplot(3, 3, 9)
                plt.imshow(cur_mask[0, 0, :, :])
                plt.pause(1)
                plt.close()
                # plt.show()
            # =============
            if epi % 200 == 0:
                torch.save(model.state_dict(), "./model_test1/modela{}_.pth".format(epi))
            # =============
            opetimizer.zero_grad()
            loss.backward()
            opetimizer.step()
            # print("epi: ", epi, "loss: ", loss)
            print("epi: ", epi, "loss: ", loss1.data + loss2.data + loss3.data)
            # torch.cuda.empty_cache()

        if i_index1 + batch_size >= train_data_size1:
            i_index1 = 0
            indices1 = np.random.permutation(train_data_size1)
        else:
            i_index1 += batch_size
        if i_index1 + 2 * batch_size >= train_data_size1:
            i_index1 = train_data_size1 - batch_size

if __name__ == "__main__":
    process()