import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

load_path = "/media/vpromise/Inter/OR/image_gray_320/S"
save_path = "/media/vpromise/Inter/OR/image_gray_320_edit/S"
for i in range(1, 243):
    edit_path = save_path + "%03d" % i + "/0/"
    os.makedirs(edit_path)
    for j in range(200):
        datapath = load_path + "%03d" % i + "/0/" + "%04d" % j + ".jpg"

        print(datapath)
        img = cv2.imread(datapath, cv2.IMREAD_GRAYSCALE)
        m = np.mean(img)
        img[img > 1.8*m] = 1.8*m

        cv2.imwrite(edit_path + "%04d" % j + ".jpg", img, )

        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()