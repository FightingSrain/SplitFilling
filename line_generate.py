
import cv2
from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def line(path):
    a = np.asarray(Image.open(path).convert('L')).astype('float')
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    # a = cv2.filter2D(a, -1, kernel=kernel)
    # cv2.imshow('ifeg', a)
    # cv2.waitKey(0)
    depth = 10.
    grad = np.gradient(a)
    grad_x, grad_y = grad

    grad_x = grad_x * depth / 100.
    grad_y = grad_y * depth / 100.

    A = np.sqrt(grad_x**2 + grad_y**2 + 1.)
    uni_x = grad_x / A
    uni_y = grad_y / A
    uni_z = 1. / A

    vec_el = np.pi / 2.2
    vec_az = np.pi / 4.
    dx = np.cos(vec_el) * np.cos(vec_az)
    dy = np.cos(vec_el) * np.sin(vec_az)
    dz = np.sin(vec_el)

    b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
    b = b.clip(0, 255)
    res = np.asarray(Image.fromarray(b.astype('uint8')))
    _, res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)  # 二值化
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    return res

path = "C://Users/10195/Desktop/AI_Painting/DanbooRegionDataset/DanbooRegion2020/trains/image/"
path_res = "C://Users/10195/Desktop/AI_Painting/DanbooRegionDataset/DanbooRegion2020/trains/line/"

file_list = glob(path + "*.png")
i = 0
for filename in file_list:
    res = line(filename)
    nums = filename.split('\\')[1].split('.')[0]
    cv2.imwrite(path_res + str(nums) + '.line.png', res)
    i += 1
    print(i)
    # plt.imshow(res)
    # plt.title('Clustered1 Image')
    # plt.show()








