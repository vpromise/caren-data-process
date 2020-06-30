import os

# save path
root_path = "/home/vpromise/Work/caren-2d-pose.pytorch/dataset/refine"

train_path = root_path + "/train_right.txt"
valid_path = root_path + "/valid_right.txt"
test_path = root_path + "/test_right.txt"

# data path 
data_path = "/media/vpromise/Inter/or/"
img_path = "img"
heatmap_path = "coor"
# os.makedirs(root_path)

# generate path txt
# with open(train_path,"w") as train_f:
#     for i in range(1,201):
#         for j in range(0, 400):
#             train_line = str(data_path + img_path + "/S" + str("%03d" % i) + "/0/" + str("%04d" % j) + ".jpg" +
#              "," + data_path + heatmap_path + "/S" + str("%03d" % i) + "/0/" + str("%04d" % j) + ".npy\n")
#             train_f.write(train_line)

with open(valid_path,"w") as valid_f:
    for i in range(201, 211):
        for j in range(0, 400, 400):
            valid_line = str(data_path + img_path + "/S" + str("%03d" % i) + "/2/" + str("%04d" % j) + ".jpg" +
             "," + data_path + heatmap_path + "/S" + str("%03d" % i) + "/2/" + str("%04d" % j) + ".npy\n")
            valid_f.write(valid_line)

# with open(test_path,"w") as test_f:
#     for i in range(211, 231):
#         for j in range(0, 400):
#             test_line = str(data_path + img_path + "/S" + str("%03d" % i) + "/0/" + str("%04d" % j) + ".jpg" +
#              "," + data_path + heatmap_path + "/S" + str("%03d" % i) + "/0/" + str("%04d" % j) + ".npy\n")
#             test_f.write(test_line)
