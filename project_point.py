import cv2
import os
import numpy as np


path1 = "/media/vpromise/Inter/or_side/coor/S"
path2 = "/media/vpromise/Inter/or_side/data/S"


for subject in range(1,227):
    save_path = path1 + "%03d" % subject + "/left_2d_1.npy"

    # save_path_0 = path1 + "%03d" % subject + "/2/"
    load_path_0 = path1+ "%03d" % subject + "/left_3d.npy"
    para_path_0 = path2 + "%03d" % subject + "/cam_para/cam_para_new_0.npy"
    
    # save_path_1 = path1 + "%03d" % subject + "/1/"
    # load_path_1 = path1 + "%03d" % subject + "/front_3d.npy"
    # para_path_1 = path2 + "%03d" % subject + "/cam_para_1.npy"
    
    # save_path_2 = path1 + "%03d" % subject + "/2/"
    # load_path_2 = path1 + "%03d" % subject + "/right_3d.npy"
    # para_path_2 = path2 + "%03d" % subject + "/cam_para_2.npy"
    
    # os.makedirs(save_path_0)

    para = np.load(para_path_0,allow_pickle=True).item()
    rvecs, tvecs, distCoeffs, cameraMatrix = para['rvecs'], para['tvecs'], para['distCoeffs'], para['cameraMatrix']

    _3d = np.load(load_path_0)
    # print(_3d.shape)
    objectPoints = _3d.reshape(-1,3)
    imgpts, jac = cv2.projectPoints(objectPoints, rvecs, tvecs, cameraMatrix, distCoeffs)
    # print(imgpts.shape)
    _2d = np.array(imgpts).reshape(-1,6)
    # print(_2d.shape)
    # for i in range(_2d.shape[0]):
    #     # print(_2d[i])  
    #     _save_path = save_path_0 + "%04d" % i + ".npy"

    #     np.save(_save_path, _2d[i])
    np.save(save_path, _2d)
    print(subject, _3d.shape, _2d.shape)