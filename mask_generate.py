


import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# path = "C://Users/10195/Desktop/AI_Painting/DanbooRegionDataset/DanbooRegion2020/trains/flat/"
# path_res = "C://Users/10195/Desktop/AI_Painting/DanbooRegionDataset/DanbooRegion2020/trains/"

path = "D://xrt/fill_dataset/flat/"
path_res = "D://xrt/fill_dataset/"

CLASS = 8

file_list = glob(path + "*.png")
i = 0
for filename in file_list:
    image = cv2.imread(filename) / 255.
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    b = np.asarray(g / (r + g + b + 1e-4) , np.float32)
    g = np.asarray(r / (r + g + b + 1e-4), np.float32)
    r = np.asarray(((r + g + b + 1e-4) / 3.) * 0.5, np.float32)
    image = np.clip(cv2.merge([r, g, b]), a_max=1., a_min=0.)
    nums = filename.split('\\')[1].split('.')[0]
    name = filename.split('\\')[1].split('.')[1]
    # plt.imshow(image)
    # plt.title('Clustered3 Image')
    # plt.show()

    image_2D = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    # Use KMeans clustering algorithm from sklearn.cluster to cluster pixels in image
    # tweak the cluster size and see what happens to the Output
    kmeans = KMeans(n_clusters=CLASS, random_state=0).fit(image_2D)
    label = kmeans.labels_.reshape(image.shape[0], image.shape[1])
    for a in range(0, CLASS):
        # label的每个像素 当为a时将 mask 中相应的位置赋值为1
        mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        mask = np.where(label == a, 1, mask)
        m = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # cv2.imwrite("./img_res/res.jpg", m)
        cv2.imwrite(path_res + "mask" + str(a) + "/" + str(nums) + '.mask.png', m)

        cv2.imshow("res4", m)  # ((image_matrixs//255)*images).astype(np.int8)
        cv2.waitKey(1)
        # plt.imshow(m)
        # plt.title('Clustered3 Image')
        # plt.show()
    i += 1
    print(i)











