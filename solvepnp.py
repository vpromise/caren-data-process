import cv2
import numpy as np
import pandas as pd

path = "/media/vpromise/Inter/Or_Dataset/images/S004/"

objectPoints = np.load(path + "right_3d.npy").reshape(-1,3)
imagePoints = np.load(path + "ann_2.npy")

k = 60

op = objectPoints[0:k]
ip = imagePoints[0:k]
print(op.shape," ",ip.shape)

# slovepnp
# camera angle 0 & 2
cameraMatrix = np.float32([[1019.7, 0, 0],
                           [0, 1021.6, 0],
                           [234.1, 303.4, 1]]).T
distCoeffs = np.float32([0.2495, 1.1288, 0, 0])

# cameraMatrix = np.float32([[827.1569, 0, 0],
#                            [0, 831.0333, 0],
#                            [214.5390, 392.4967, 1]]).T
# distCoeffs = np.float32([-0.2310, 0.5603, 0, 0])

_, rvecs, tvecs = cv2.solvePnP(op, ip, cameraMatrix, distCoeffs)
print("rvecs = \n", rvecs)
print("tvecs = \n", tvecs)


imgpts, jac = cv2.projectPoints(objectPoints, rvecs, tvecs, cameraMatrix, distCoeffs)

err = ((imgpts.reshape(-1, 2))[0:imagePoints.shape[0]] - imagePoints)
average_err = sum(sum(abs(err)/err.shape[0]))
print("average err = ", average_err)
print(err)

# save
cam_para = {'cameraMatrix':cameraMatrix, 'distCoeffs':distCoeffs, 
              'rvecs':rvecs, 'tvecs':tvecs, 'average_err':average_err}
np.save(path + 'cam_para_2.npy', cam_para) 


read_dictionary = np.load(path +'cam_para_2.npy',allow_pickle=True).item()

print(read_dictionary['rvecs']) 
print(read_dictionary['tvecs']) 
print(read_dictionary['distCoeffs']) 
print(read_dictionary['cameraMatrix'])
print(read_dictionary['average_err'])