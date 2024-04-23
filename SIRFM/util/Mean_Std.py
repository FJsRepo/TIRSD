import os
import cv2
import numpy as np
# from imageio import imread

filepath = '/home/wacht/1-Transformer/MTD/train_images/'  # the path of dataset
pathDir = os.listdir(filepath)

# 自己的方法 最终采用自己方法的结果
R_mean = 0
G_mean = 0
B_mean = 0
R_std = 0
G_std = 0
B_std = 0
totalImgNum = len(pathDir)
for idx in range(totalImgNum):
    filename = pathDir[idx]
    if filename[0] != '.':
        img = cv2.imread(os.path.join(filepath, filename))
        img = img / 255.0
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        R_mean += np.mean(R)
        G_mean += np.mean(G)
        B_mean += np.mean(B)
        R_std += np.std(R)
        G_std += np.std(G)
        B_std += np.std(B)

R_mean = R_mean / totalImgNum
G_mean = G_mean / totalImgNum
B_mean = B_mean / totalImgNum
R_std = R_std / totalImgNum
G_std = G_std / totalImgNum
B_std = B_std / totalImgNum

# [R_mean, G_mean, B_mean]: 0.49115652408776983 0.49115652408776983 0.49115652408776983
# [R_std, G_std, B_std] 0.2278842012782631 0.2278842012782631 0.2278842012782631
# [R_mean, G_mean, B_mean]: 0.491 0.491 0.491
# [R_std, G_std, B_std] 0.228 0.228 0.228
print('[R_mean, G_mean, B_mean]:', round(R_mean, 3), round(G_mean, 3), round(B_mean, 3))
print('[R_std, G_std, B_std]', round(R_std, 3), round(G_std, 3), round(B_std, 3))



# Github 方法
# R_mean is 0.491783, G_mean is 0.491783, B_mean is 0.491783
# R_var is 0.230345, G_var is 0.230345, B_var is 0.230345
# R_mean is 0.492000, G_mean is 0.492000, B_mean is 0.492000
# R_var is 0.230000, G_var is 0.230000, B_var is 0.230000

# R_channel = 0
# G_channel = 0
# B_channel = 0
# img_size = 0
#
# for idx in range(len(pathDir)):
#     filename = pathDir[idx]
#     if filename[0] != '.':
#         img = cv2.imread(os.path.join(filepath, filename))
#         img = img / 255.0
#         imgHeight = img.shape[0]
#         imgWidth = img.shape[1]
#         img_size = img_size + imgHeight * imgWidth
#         R_channel = R_channel + np.sum(img[:, :, 0])
#         G_channel = G_channel + np.sum(img[:, :, 1])
#         B_channel = B_channel + np.sum(img[:, :, 2])
#
# R_mean = R_channel / img_size
# G_mean = G_channel / img_size
# B_mean = B_channel / img_size
#
# R_channel = 0
# G_channel = 0
# B_channel = 0
# for idx in range(len(pathDir)):
#     filename = pathDir[idx]
#     if filename[0] != '.':
#         img = cv2.imread(os.path.join(filepath, filename))
#         img = img / 255.0
#         R_channel = R_channel + np.sum((img[:, :, 0] - R_mean)**2)
#         G_channel = G_channel + np.sum((img[:, :, 1] - G_mean)**2)
#         B_channel = B_channel + np.sum((img[:, :, 2] - B_mean)**2)
#
# R_var = (R_channel / img_size)**0.5
# G_var = (G_channel / img_size)**0.5
# B_var = (B_channel / img_size)**0.5
#
# print("R_mean is %f, G_mean is %f, B_mean is %f" % (round(R_mean, 3), round(G_mean, 3), round(B_mean, 3)))
# print("R_var is %f, G_var is %f, B_var is %f" % (round(R_var, 3), round(G_var, 3), round(B_var, 3)))
