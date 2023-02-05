from glob import glob
import os
import time
import cv2
import numpy as np
from flat_image import vis
from influence import main

path1 = "C://Users/10195/Desktop/AI_Painting/DanbooRegionDataset/DanbooRegion2020/trains/image/"
path2 = "C://Users/10195/Desktop/AI_Painting/DanbooRegionDataset/DanbooRegion2020/trains/region/"
path3 = "C://Users/10195/Desktop/AI_Painting/DanbooRegionDataset/DanbooRegion2020/trains/flat/"
def read_file_lst(path):
    file_list = glob(path + "*.png")
    # print(file_list)
    img_lst = []
    # region_lst = []
    i = 1
    for filename in file_list:
        # image = cv2.imread(filename)
        nums = filename.split('\\')[1].split('.')[0]
        name = filename.split('\\')[1].split('.')[1]
        # print(nums)
        # if name == "image":
        img_lst.append(filename)
        # elif name == "region":
        #     region_lst.append(filename)
            # cv2.imshow("res", image)
            # cv2.waitKey(0)
        # if not os.path.exists(path + nums + 'image.png'):
        #     t1 = time.time()
        #     skeleton, region, flatten = segment(image)
        #     t2 = time.time()
        #     print(t2 - t1)
        #     cv2.imwrite('C:/Users/10195/Desktop/sketch_pairs/Approximate professional NPS/sketch/' + nums + 'A_NPSsketch.png', skeleton)
        #     cv2.imwrite('C:/Users/10195/Desktop/sketch_pairs/Approximate professional NPS/region/' + nums + 'A_NPSregion.png', region)
        #
        # print("num:" + str(i), 'current complete: ' + nums)
        i += 1
    return img_lst

def generate_flat_color(img_lst, region_lst):
    i = 1
    for f1, f2 in zip(img_lst, region_lst):
        nums = f1.split('\\')[1].split('.')[0]
        img = cv2.imread(f1)
        reg = cv2.imread(f2)
        flat_img = vis(reg, img)
        cv2.imshow("res", flat_img)
        cv2.waitKey(1)
        cv2.imwrite(path3 + nums + '.flat.png', flat_img)
        i += 1
        print(i)

# img_list = read_file_lst(path1)
# region_list = read_file_lst(path2)
#
# generate_flat_color(img_list, region_list)
# file_list = glob(path3 + "*.png")
# print(file_list)
# for i in file_list:
#     # name1 = i.split('\\')[1].split('.')[0][-4:]
#     # name2 = i.split('\\')[1].split('.')[0][:-4]
#     nums = i.split('\\')[1].split('.')[0][:-4]
#     print(nums)
#     os.rename(i, path3 + nums + '.flat.png')
#     # print(name1, name2)
#     # cv2.imwrite(path3 + nums + '.flat.png', flat_img)
# print(read_file_lst(path2))
file_list = glob(path3 + "*.png")
i = 0
for filename in file_list:
    image = cv2.imread(filename)
    nums = filename.split('\\')[1].split('.')[0]
    name = filename.split('\\')[1].split('.')[1]
    # print(name, nums)
    # print(nums)
    main(image, nums, nbr_class=8, iterations=5)
    i += 1
    print(i)