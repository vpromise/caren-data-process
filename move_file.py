import os
import shutil

root_path = "/media/vpromise/Inter/or_side/coor/"
save_path = "/media/vpromise/Inter/orthopedic/2d_data/coor"

file_count = 0

subjects = os.listdir(root_path)
for subject in subjects:

    # load data
    subject_path = os.path.join(root_path, subject)
    data_path = subject_path + "/left_2d.npy"

    # save data
    # os.makedirs(os.path.join(save_path, subject))
    dst_path = os.path.join(save_path, subject)
    
    print("Coping file:", data_path)
    print("Saving to:", dst_path)
    shutil.copy2(data_path, dst_path)
    print(" ")

    file_count += 1

print("Processed with %d files!" % file_count)