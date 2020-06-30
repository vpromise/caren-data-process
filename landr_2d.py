import os
import shutil
import numpy as np


root_path = "/media/vpromise/Inter/orthopedic/3d_data_seperate/coor_2d"

subjects = os.listdir(root_path)
for subject in subjects:
    path = os.path.join(root_path, subject)
    left_path = path + "/left_2d.npy"
    right_path = path + "/right_2d.npy"

    save_path = path + "/all/"
    # os.makedirs(save_path)

    left = np.load(left_path)
    right = np.load(right_path)
    print("Processing with subject:", subject, left.shape, right.shape)

    for i in range(left.shape[0]):
        l_and_r = np.vstack((left[i][np.newaxis,:,:],right[i][np.newaxis,:,:]))
        _save_path = save_path + "%04d" % i + ".npy"
        np.save(_save_path, l_and_r)


