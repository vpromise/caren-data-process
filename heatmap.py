import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt

savepath = "/media/vpromise/Inter/or/hm_sigma_5/S"
parapath = "/media/vpromise/Inter/or/coor/S"

# def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
#     X1 = np.linspace(1, img_width, img_width)
#     Y1 = np.linspace(1, img_height, img_height)
#     [X, Y] = np.meshgrid(X1, Y1)
#     X = X - c_x
#     Y = Y - c_y
#     D2 = X * X + Y * Y
#     E2 = 2.0 * sigma * sigma
#     Exponent = D2 / E2
#     heatmap = np.exp(-Exponent)
#     return heatmap

# heatmap = CenterLabelHeatMap(640, 480, 300, 200, 20)
# [u, v] = [int(np.argmax(heatmap))//480,  int(np.argmax(heatmap))%480]
# print("[u, v] = ", [u, v])
# print(heatmap.shape)
# plt.imshow(heatmap)
# plt.show()

def CenterLabelHeatMap(img_height, img_width, c_x, c_y, sigma):
    X1 = np.linspace(1, img_height, img_height)
    Y1 = np.linspace(1, img_width, img_width)
    [Y, X] = np.meshgrid(Y1, X1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap

for subject in range(1,235):

    save_path = savepath + "%03d" % subject + "/0/"
    os.makedirs(save_path)
    load_path = parapath + "%03d" % subject + "/left_2d.npy"

    _2d = np.load(load_path).reshape(-1,2)

    hm = []
    for points in range(0, _2d.shape[0]):
        [c_y, c_x] = _2d[points]
        # c_x = (c_x - 280)/4
        # c_y = (c_y - 20)/4 # left
        c_y = (c_y - 110)/4 # right
        
        # front
        # c_x = (c_x - 120)/4
        # c_y = (c_y - 110)/4

        # print([c_x, c_y])

        hm.append(CenterLabelHeatMap(120, 80, c_x, c_y, 5))

    hm = np.array(hm).reshape(400,5,120,80)

    print("subject is ", subject)
    print(hm.shape)
    print(save_path)
    # plt.imshow(hm[1][1])
    # plt.show()
    for number in range(0, hm.shape[0]):

        hm_savepath = save_path + "%04d" % number + ".npy"
        # print(hm_savepath)

        np.save(hm_savepath, np.array(hm[number]))