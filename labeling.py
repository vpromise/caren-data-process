import os
import csv
import cv2
import pandas as pd
import numpy as np

kk = 20
jj = 3

data_path = "/media/vpromise/Inter/OR_DATA/cam_cali/S165/"
# '''
a = np.zeros((1,2))
np.save(data_path + "/ann_2.npy", a)

def click(event, x, y, flags, param):
    global img, img_copy
    font = cv2.FONT_HERSHEY_PLAIN

    if event==cv2.EVENT_LBUTTONDOWN:
        temp = str(x)+ ',' + str(y)
        cv2.circle(img, (x, y), 3, (0, 0, 255), 1)
        cv2.putText(img, temp, (x, y), font, 1.5, (0,255,0), 1)
        point = np.array([x, y])
#       print(point)
        annotation = np.load(data_path + "ann_2.npy")
        annotation = np.vstack((annotation, point))
        np.save(data_path + "ann_2.npy", annotation)
#       print("point saved")

for image_count in range(0, kk):
    path = data_path +"2/" + str("%04d" % (int(image_count))) + ".jpg"
    print(path)
    img = cv2.imread(path)
    img_copy = img.copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback('image', click)

    image_count += 1
    
    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('q'):
            
            img = img_copy.copy()
            img_copy = img.copy()
    cv2.destroyAllWindows()

# save annotation
a = np.load(data_path + "ann_2.npy")
np.save(data_path + "ann_2.npy", a[1::])
print("congratulate!!! 2d label saved !!! for data " + data_path[-5:-1])

# '''
path = data_path

objectPoints = np.load(path + "right_3d.npy").reshape(-1,3)
print(objectPoints.shape)
imagePoints = np.load(path + "ann_2.npy")

op = objectPoints[0:kk*jj:]
ip = imagePoints[0:kk*jj]
print(op.shape," ",ip.shape)

# slovepnp
# camera angle 1

# cameraMatrix = np.float32([[865.5935, 0, 0],[0, 866.3518, 0],[231.6897, 334.0235, 1]]).T
# distCoeffs = np.float32([-0.1111, 0.8023, 0, 0])

# cameraMatrix = np.float32([[232.6503, 0, 0], [0, 229.9091, 0], [260.3345, 279.009, 1.0000]]).T
# distCoeffs = np.float32([-0.1388, 0.1774, 0, 0])

# camera angle 0 & 2
# cameraMatrix = np.float32([[115.5821, 0, 0], [0, 114.2162, 0], [260.5226, 272.8565, 1.0000]]).T      44
# distCoeffs = np.float32([-0.0365, 0.0132, 0, 0])
# cameraMatrix = np.float32([[32.2344, 0, 0], [0, 31.8535, 0], [260.5319, 272.4498, 1.0000]]).T       119
# distCoeffs = np.float32([-0.0028, 8.0339e-05, 0, 0])
# cameraMatrix = np.float32([[133.4631, 0, 0], [0, 131.8723, 0], [260.7056, 273.3678, 1.0000]]).T     41
# distCoeffs = np.float32([-0.0485, 0.0233, 0, 0])
# cameraMatrix = np.float32([[217.8670, 0, 0], [0, 215.2657, 0], [260.61225, 276.9998, 1.0000]]).T   31
# distCoeffs = np.float32([-0.1240, 0.1458, 0, 0])
# 29
# cameraMatrix = np.float32([[232.6503, 0, 0], [0, 229.9091, 0], [260.3345, 279.009, 1.0000]]).T
# distCoeffs = np.float32([-0.1388, 0.1774, 0, 0])
# cameraMatrix = np.float32([[124.5057, 0, 0], [0, 123.0272, 0], [259.9831, 276.6895, 1.0000]]).T
# distCoeffs = np.float32([-0.0410, 0.0159, 0, 0])
# cameraMatrix = np.float32([[21.8447, 0, 0], [0, 21.5914, 0], [260.1948, 281.7818, 1.0000]]).T
# distCoeffs = np.float32([-0.0011, 1.1074e-05, 0, 0])
# cameraMatrix = np.float32([[65.9546, 0, 0], [0, 65.1778, 0], [259.6670, 278.7286, 1.0000]]).T
# distCoeffs = np.float32([-0.0110, 0.0011, 0, 0])
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
average_err = (np.sum(np.sqrt((np.sum(np.square(err),1))))/err.shape[0])
# average_err = sum(sum(abs(err)))/err.shape[0]
print("average err = ", average_err)


# save
cam_para = {'cameraMatrix':cameraMatrix, 'distCoeffs':distCoeffs, 
              'rvecs':rvecs, 'tvecs':tvecs, 'average_err':average_err}
np.save(path + 'cam_para_22.npy', cam_para) 


read_dictionary = np.load(path +'cam_para_22.npy',allow_pickle=True).item()

print(read_dictionary['rvecs']) 
print(read_dictionary['tvecs']) 
print(read_dictionary['distCoeffs']) 
print(read_dictionary['cameraMatrix'])
print(read_dictionary['average_err'])