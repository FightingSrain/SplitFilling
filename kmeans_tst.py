


import cv2
import numpy as np

import matplotlib.pyplot as plt
img = cv2.imread("./img_test/3.png") / 255
# enhancer
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]
b = np.asarray(g / (r + g + b))
g = np.asarray(r / (r + g + b))
r = np.asarray(((r + g + b)/3) * 0.5)
img = cv2.merge([r, g, b])

image_2D = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
# Use KMeans clustering algorithm from sklearn.cluster to cluster pixels in image
from sklearn.cluster import KMeans
# tweak the cluster size and see what happens to the Output
kmeans = KMeans(n_clusters=3, random_state=0).fit(image_2D)
print(kmeans.labels_.reshape(img.shape[0], img.shape[1]))
print(kmeans.labels_.max())
print(kmeans.labels_.min())
label = kmeans.labels_.reshape(img.shape[0], img.shape[1])
# mask = np.zeros((img.shape[0], img.shape[1]))
plt.imshow(label)
plt.title('Clustered4 Image')
plt.show()
for a in range(0, 3):
    # label的每个像素 当为a时将 mask 中相应的位置赋值为1
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask = np.where(label == a, 1, mask)
    m = cv2.cvtColor((mask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imwrite("./img_res/res.jpg", m)
    cv2.imshow("res", m)  # ((image_matrixs//255)*images).astype(np.int8)
    cv2.waitKey(0)
    print(mask)
    plt.imshow(mask)
    plt.title('Clustered3 Image')
    plt.show()
plt.imshow(kmeans.labels_.reshape(img.shape[0], img.shape[1]))
plt.title('Clustered2 Image')
plt.show()

clustered = kmeans.cluster_centers_[kmeans.labels_]
# Reshape back the image from 2D to 3D image
clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
plt.imshow(clustered_3D)
plt.title('Clustered Image')
plt.show()
