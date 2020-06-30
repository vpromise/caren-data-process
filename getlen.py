import cv2
import os
import csv
import numpy as np

'''
Gets the length of videos and csv
'''

def csv_process_fn(csv_file, csv_full_path):
    csv_leng = 0
    for csv_file_name in csv_file:
        file_path = os.path.join(csv_full_path, csv_file_name)
        with open(file_path, 'r') as f:
            csv_leng = len(f.readlines())
    return csv_leng
def video_process_fn(video_file, video_full_path):
    for video_file_name in video_file:
        file_path = os.path.join(video_full_path, video_file_name)
        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            capture.open(file_path)
            print('cap is not opened.')
        frame = capture.get(7)
    return frame

def get_len(root_path):
    total_dataset_num = 0
    val_num = 0

    dates = os.listdir(root_path)
    for date in dates:
        full_path = os.path.join(root_path, date)
        files = os.listdir(full_path)
        print("dir is " + full_path)
        csv_file = filter(lambda x: x.endswith('.csv'), files)
        csv_len = csv_process_fn(csv_file, full_path)
        video_file_0 = filter(lambda x: x.endswith('0.avi'), files)
        video_len = video_process_fn(video_file_0, full_path)
        print(csv_len//100 , video_len/50)
        if video_len/50 < 50 :
            print("********************attention*********************\n")
        if csv_len//100 < 50:
            print("********************attention*********************\n")
                
    return csv_len, video_len


# print("train dataset")
get_len('/media/vpromise/Inter/Or_Dataset/data_full')
