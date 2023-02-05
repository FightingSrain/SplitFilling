


import numpy as np
import matplotlib.pyplot as plt

a1 = np.zeros((10, 10))
a1[5, 3:7] = 1

a2 = np.zeros((10, 10))
a2[2:6, 3] = 1

a3 = np.zeros((10, 10))
a3[8:9, 8:9] = 1

#-------
value = 1
a_sum = a1 + a2 + a3
# res = np.ones_like(a1) * value
u = np.clip(a_sum - a1, a_min=0, a_max=1)
res = np.where(u == 1, -value, 0)
res = res * (1 - a1) + a1 * value



# res = np.where(a1 == 1, res, a1)
import cv2
cv2.line(res, (0, 0), (1, 1), (1, 1), 1, 8)
plt.imshow(res)
plt.title('Clustered3 Image')
plt.show()

# plt.imshow((np.clip(a2 + a3, a_min=0, a_max=1)))
# plt.title('Clustered3 Image')
# plt.show()

# plt.imshow(a1)
# plt.title('Clustered1 Image')
# plt.show()
# plt.imshow(a2)
# plt.title('Clustered2 Image')
# plt.show()
# plt.imshow(a3)
# plt.title('Clustered3 Image')
# plt.show()