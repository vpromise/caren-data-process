import os
import shutil
import numpy as np


root_path = "/media/vpromise/Inter/orthopedic/3d_data_cons/coor_3d"


file_count = 0

subjects = os.listdir(root_path)
for subject in subjects:
    path = os.path.join(root_path, subject)
    data_path = path +  "/all_3d.npy"

    save_path = os.path.join(path, 'all')
    os.makedirs(save_path)

    _2d = np.load(data_path).reshape(-1,8,3)

    for i in range(_2d.shape[0]):
        save_file_path = save_path + "/" + "%04d" % i + ".npy"
        np.save(save_file_path, _2d[i])

    print(subject, _2d.shape)