import cv2
import numpy as np
import matplotlib.pylab as plt
import random
def get_euclidian_distance(vec1, vec2):

    return np.linalg.norm(vec1 - vec2)


def get_centroids(centroids, vector):
    minimum_distance = float('inf')
    centroid_index = -1
    for index, centroid in enumerate(centroids):

        distance = get_euclidian_distance(centroid, vector)
        if distance < minimum_distance:
            minimum_distance = distance
            centroid_index = index

    return centroid_index + 1

def k_means(vector_set, centroid_num, iteration_times=20):

    group_tag = np.zeros(len(vector_set))
    centroids = []
    #初始化聚类中心
    for i in range(centroid_num):

        #r_channel = random.randint(0, 255)
        #g_channel = random.randint(0, 255)
        #b_channel = random.randint(0, 255)

        a = random.randint(0, len(vector_set))

        centroid = np.array(vector_set[a])

        centroids.append(centroid)

    for j in range(iteration_times):
        print("开始第{}轮聚类".format(str(j)))

        #开始一轮新的分类
        for index, vector in enumerate(vector_set):

            centroid_index = get_centroids(centroids, vector)

            group_tag[index] = centroid_index

        #重新寻找聚类中心
        centroid_sum = np.zeros((centroid_num, 3))  # 记录属于每个中心的向量的和
        vector_nbr = np.zeros(centroid_num)  # 记录属于每个向量中心的向量的个数
        for index, vector in enumerate(vector_set):

            group = int(group_tag[index] - 1)

            temp = centroid_sum[int(group), :]
            temp = temp + np.array(vector)

            centroid_sum[group] = temp
            vector_nbr[group] += 1

        for k in range(centroid_num):

            centroid_sum[k] /= vector_nbr[k]

        centroids = centroid_sum
    return group_tag, centroids


path = "C://Users/10195/Desktop/AI_Painting/DanbooRegionDataset/DanbooRegion2020/trains/"

def main(images, num, nbr_class=3, iterations=5):
    # image = cv2.resize(images, (512, 512))
    w, h, c = images.shape
    pixel_set = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            pixel_vector = image[i, j, :]
            pixel_set.append(pixel_vector)

    group_tag, centroids = k_means(pixel_set, nbr_class, iterations)

    group_tag = group_tag.reshape((image.shape[0], image.shape[1]))



    for k in range(nbr_class):

        image_matrix = np.zeros((image.shape[0], image.shape[1], 3))

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                c = group_tag[i][j] - 1
                if k == group_tag[i][j] - 1:
                    image_matrix[i, j, :] = image[i, j, :] # 255  # image[i, j, :]
        # print(str(path + "mask" + str(k) + "/" + str(num) + '.mask.png'))
        # cv2.imwrite(path + "mask" + str(k) + "/" + str(num) + '.mask.png', image_matrix) # **
        # print(image_matrix)
        # image_matrixs = cv2.medianBlur(((image_matrix)).astype(np.uint8), 5)
        image_matrixs = cv2.resize(image_matrix, (h, w))
        # if image_matrixs > 0:
        #     image_matrixs = 1
        # else:
        #     image_matrixs = 0
        print(w, h)
        print(image_matrix.shape)
        print("++++++++++")

        cv2.imshow("res", (image_matrixs*255).astype(np.uint8))  # ((image_matrixs//255)*images).astype(np.int8)
        cv2.waitKey(0)

    # total_row = nbr_class / 2
    # plt.subplot(total_row + 1, 2, 1)
    # plt.imshow(image)
    # plt.axis("off")
    # plt.title("original image")
    # for i in range(nbr_class):
    #
    #     row = i / 3
    #     col = i % 3
    #     image_matrix = np.load("{}.npy".format(str(int(i))))
    #     plt.subplot(total_row + 1 , 2, i + 2)
    #     plt.axis("off")
    #     plt.title("part{}".format(str(int(i+1))))
    #     plt.imshow(image_matrix / 255)
    # plt.show()




if __name__ == '__main__':

    image = cv2.imread("./img_test/1.jpg")/255
    # image = cv2.resize(image, (512, 512))
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    # b = np.asarray(g / (r + g + b)+1, np.uint8)
    # g = np.asarray(r / (r + g + b)+1, np.uint8)
    # r = np.asarray(((r + g + b)/3) * 10, np.uint8)
    image = cv2.merge([r, g, b])
    # b, g, r = cv2.split(tmp)
    # # print(b.shape)
    # bs = g.mean() / (r.mean() + g.mean() + b.mean())
    # gs = r.mean() / (r.mean() + g.mean() + b.mean())
    # rs = ((r.mean() + g.mean() + b.mean()) / 3) * 0.5
    plt.imshow(image)
    plt.show()
    import time
    t1 = time.time()
    main(image, 1, nbr_class=6, iterations=2)
    t2 = time.time()
    print((t2 - t1))