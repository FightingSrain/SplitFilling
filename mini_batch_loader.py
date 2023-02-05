import os
import numpy as np
import cv2
import torch

class MiniBatchLoader(object):

    def __init__(self, flat_img_path, mask0_img_path, mask1_img_path, mask2_img_path, mask3_img_path,
                 mask4_img_path, mask5_img_path, mask6_img_path, mask7_img_path,
                 skeleton_img_path, line_img_path, image_dir_path, crop_size):

        # load data paths
        self.flat_img_path_infos = self.read_paths(flat_img_path, image_dir_path) # flat img
        self.mask0_img_path_infos = self.read_paths(mask0_img_path, image_dir_path)  # mask0 img
        self.mask1_img_path_infos = self.read_paths(mask1_img_path, image_dir_path)  # mask1 img
        self.mask2_img_path_infos = self.read_paths(mask2_img_path, image_dir_path)  # mask2 img
        self.mask3_img_path_infos = self.read_paths(mask3_img_path, image_dir_path)  # mask3 img
        self.mask4_img_path_infos = self.read_paths(mask4_img_path, image_dir_path)  # mask4 img
        self.mask5_img_path_infos = self.read_paths(mask5_img_path, image_dir_path)  # mask5 img
        self.mask6_img_path_infos = self.read_paths(mask6_img_path, image_dir_path)  # mask6 img
        self.mask7_img_path_infos = self.read_paths(mask7_img_path, image_dir_path)  # mask7 img

        self.skeleton_img_path_infos = self.read_paths(skeleton_img_path, image_dir_path)  # skeleton img
        self.line_img_path_infos = self.read_paths(line_img_path, image_dir_path)  # line img

        # self.training_target_path_infos = self.read_paths(train_target_path, image_dir_path)

        # test
        # self.testing_path_infos = self.read_paths(test_path, image_dir_path)

        self.crop_size = crop_size

    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path

    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c

    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs

    def load_training_data(self, indices):
        return self.load_data(self.flat_img_path_infos,
                              self.mask0_img_path_infos,
                              self.mask1_img_path_infos,
                              self.mask2_img_path_infos,
                              self.mask3_img_path_infos,
                              self.mask4_img_path_infos,
                              self.mask5_img_path_infos,
                              self.mask6_img_path_infos,
                              self.mask7_img_path_infos,
                              self.skeleton_img_path_infos,
                              self.line_img_path_infos, indices, augment=True)

    # def load_testing_data(self, indices):
    #     return self.load_data(self.testing_path_infos, self.training_target_path_infos, indices)

    # test ok
    def load_data(self, flat_path_infos, mask0_path_infos, mask1_path_infos, mask2_path_infos, mask3_path_infos,
                  mask4_path_infos, mask5_path_infos, mask6_path_infos, mask7_path_infos,
                  skeleton_path_infos, line_path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 3
        # tr_size = np.random.randint(100, 256)
        tr_size = 256
        if augment:
            flats = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            masks0 = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            masks1 = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            masks2 = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            masks3 = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            masks4 = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            masks5 = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            masks6 = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            masks7 = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            skeletons = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            lines = np.zeros((mini_batch_size, in_channels, tr_size, tr_size)).astype(np.float32)
            for i, index in enumerate(indices):
                flat_path = flat_path_infos[index]
                mask0_path = mask0_path_infos[index]
                mask1_path = mask1_path_infos[index]
                mask2_path = mask2_path_infos[index]
                mask3_path = mask3_path_infos[index]
                mask4_path = mask4_path_infos[index]
                mask5_path = mask5_path_infos[index]
                mask6_path = mask6_path_infos[index]
                mask7_path = mask7_path_infos[index]
                skeleton_path = skeleton_path_infos[index]
                line_path = line_path_infos[index]
                #---------------
                flat = cv2.imread(flat_path)
                mask0 = cv2.imread(mask0_path)
                mask1 = cv2.imread(mask1_path)
                mask2 = cv2.imread(mask2_path)
                mask3 = cv2.imread(mask3_path)
                mask4 = cv2.imread(mask4_path)
                mask5 = cv2.imread(mask5_path)
                mask6 = cv2.imread(mask6_path)
                mask7 = cv2.imread(mask7_path)
                skeleton = cv2.imread(skeleton_path)
                line = cv2.imread(line_path, cv2.IMREAD_UNCHANGED)

                # 随机缩放
                # ind_s = np.random.randint(1, 4)
                ind_s = 1
                h, w, c = flat.shape
                flat = cv2.resize(flat, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)
                mask0 = cv2.resize(mask0, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)
                mask1 = cv2.resize(mask1, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)
                mask2 = cv2.resize(mask2, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)
                mask3 = cv2.resize(mask3, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)
                mask4 = cv2.resize(mask4, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)
                mask5 = cv2.resize(mask5, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)
                mask6 = cv2.resize(mask6, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)
                mask7 = cv2.resize(mask7, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)
                skeleton = cv2.resize(skeleton, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)
                line = cv2.resize(line, (w // ind_s, h // ind_s), interpolation=cv2.INTER_AREA)

                h, w, c = flat.shape

                # 数据增强
                if np.random.rand() > 0.5:
                    flat = np.fliplr(flat)
                    mask0 = np.fliplr(mask0)
                    mask1 = np.fliplr(mask1)
                    mask2 = np.fliplr(mask2)
                    mask3 = np.fliplr(mask3)
                    mask4 = np.fliplr(mask4)
                    mask5 = np.fliplr(mask5)
                    mask6 = np.fliplr(mask6)
                    mask7 = np.fliplr(mask7)
                    skeleton = np.fliplr(skeleton)
                    line = np.fliplr(line)
                if np.random.rand():
                    angle = 10*np.random.rand()
                    if np.random.rand()>0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                    flat = cv2.warpAffine(flat, M, (w, h))
                    mask0 = cv2.warpAffine(mask0, M, (w, h))
                    mask1 = cv2.warpAffine(mask1, M, (w, h))
                    mask2 = cv2.warpAffine(mask2, M, (w, h))
                    mask3 = cv2.warpAffine(mask3, M, (w, h))
                    mask4 = cv2.warpAffine(mask4, M, (w, h))
                    mask5 = cv2.warpAffine(mask5, M, (w, h))
                    mask6 = cv2.warpAffine(mask6, M, (w, h))
                    mask7 = cv2.warpAffine(mask7, M, (w, h))
                    skeleton = cv2.warpAffine(skeleton, M, (w, h))
                    line = cv2.warpAffine(line, M, (w, h))


                rand_range_h = h - self.crop_size
                rand_range_w = w - self.crop_size

                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)

                flat_t = flat[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                mask0_t = mask0[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                mask1_t = mask1[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                mask2_t = mask2[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                mask3_t = mask3[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                mask4_t = mask4[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                mask5_t = mask5[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                mask6_t = mask6[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                mask7_t = mask7[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]


                skeleton_t = skeleton[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                line_t = line[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]

                # 随机旋转
                # angle = np.random.randint(0, 3)
                # if angle == 0:
                #     flat_t = np.rot90(flat_t, 1)
                #     mask0_t = np.rot90(mask0_t, 1)
                #     mask1_t = np.rot90(mask1_t, 1)
                #     mask2_t = np.rot90(mask2_t, 1)
                #     mask3_t = np.rot90(mask3_t, 1)
                #     mask4_t = np.rot90(mask4_t, 1)
                #     skeleton_t = np.rot90(skeleton_t, 1)
                #     line_t = np.rot90(line_t, 1)
                # if angle == 1:
                #     flat_t = np.rot90(flat_t, 2)
                #     mask0_t = np.rot90(mask0_t, 2)
                #     mask1_t = np.rot90(mask1_t, 2)
                #     mask2_t = np.rot90(mask2_t, 2)
                #     mask3_t = np.rot90(mask3_t, 2)
                #     mask4_t = np.rot90(mask4_t, 2)
                #     skeleton_t = np.rot90(skeleton_t, 2)
                #     line_t = np.rot90(line_t, 2)
                # if angle == 2:
                #     flat_t = np.rot90(flat_t, 3)
                #     mask0_t = np.rot90(mask0_t, 3)
                #     mask1_t = np.rot90(mask1_t, 3)
                #     mask2_t = np.rot90(mask2_t, 3)
                #     mask3_t = np.rot90(mask3_t, 3)
                #     mask4_t = np.rot90(mask4_t, 3)
                #     skeleton_t = np.rot90(skeleton_t, 3)
                #     line_t = np.rot90(line_t, 3)

                # 缩放
                flats[i, :, :, :] = cv2.resize(np.asarray(flat_t),
                                      (tr_size, tr_size),
                                      interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.
                # 随机色彩偏移
                alpha = 0.2
                b1, g1, r1 = flats[i, 0:1, :, :], flats[i, 1:2, :, :], flats[i, 2:3, :, :]
                # b_w = np.random.uniform(0.114 - alpha, 0.114 + alpha)
                # g_w = np.random.uniform(0.587 - alpha, 0.587 + alpha)
                # r_w = np.random.uniform(0.299 - alpha, 0.299 + alpha)
                # out1 = (b_w * b1 + g_w * g1 + r_w * r1) / (b_w + g_w + r_w)

                b_w = np.random.rand()*2
                g_w = np.random.rand()*2
                r_w = np.random.rand()*2

                flats[i, :, :, :] = np.clip(np.concatenate([b_w * b1, g_w * g1, r_w * r1], 0), a_max=1., a_min=0.)


                masks0[i, :, :, :] = cv2.resize(np.asarray(mask0_t),
                                               (tr_size, tr_size),
                                               interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.
                masks1[i, :, :, :] = cv2.resize(np.asarray(mask1_t),
                                               (tr_size, tr_size),
                                               interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.
                masks2[i, :, :, :] = cv2.resize(np.asarray(mask2_t),
                                               (tr_size, tr_size),
                                               interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.
                masks3[i, :, :, :] = cv2.resize(np.asarray(mask3_t),
                                               (tr_size, tr_size),
                                               interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.
                masks4[i, :, :, :] = cv2.resize(np.asarray(mask4_t),
                                               (tr_size, tr_size),
                                               interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.
                masks5[i, :, :, :] = cv2.resize(np.asarray(mask5_t),
                                                (tr_size, tr_size),
                                                interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.
                masks6[i, :, :, :] = cv2.resize(np.asarray(mask6_t),
                                                (tr_size, tr_size),
                                                interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.
                masks7[i, :, :, :] = cv2.resize(np.asarray(mask7_t),
                                                (tr_size, tr_size),
                                                interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.

                skeletons[i, :, :, :] = cv2.resize(np.asarray(skeleton_t),
                                               (tr_size, tr_size),
                                               interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.
                lines[i, :, :, :] = cv2.resize(np.asarray(line_t),
                                            (tr_size, tr_size),
                                            interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.

        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        return flats, masks0, masks1, masks2, masks3, masks4, masks5, masks6, masks7, skeletons, lines
