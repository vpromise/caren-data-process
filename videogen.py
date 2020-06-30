# video process function
# extract frames from videos
import os
import cv2
import numpy as np



def video2frame(video_file, video_path, img_path):
    '''
    video to frame image
    start at : start_frame
    end at : end_frame
    extract every frame : skip_frame
    '''
    start_time = 10
    end_time = 30
    skip_frame = 1

    frequency = 50
    start_frame = frequency * start_time
    end_frame = frequency * end_time


    os.makedirs(img_path)

    for video in video_file:
        video_full_path = os.path.join(video_path, video)
        print("Process video : ", video_full_path)
        capture = cv2.VideoCapture(video_full_path)

        frame_count = 0
        img_count = 0

        success, frame = capture.read()
        while (success):
            if (start_frame <= frame_count < end_frame and frame_count % skip_frame == 0):
                frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) # save image as gray type
                # print("save frame count : ", frame_count)
                frame = frame[120:600, 20:340] # crop_left
                # frame = frame[10:550, 60:420] # crop_front [540,360]
                # frame = frame[120:600, 110:430] # crop_right
                m = np.mean(frame)
                frame[frame > 1.8*m] = 1.8*m
                cv2.imwrite(img_path + "/" + "%04d" % img_count + ".jpg", frame)
                img_count += 1
            success, frame = capture.read()
            frame_count += 1
        print("extract %d frames" % (img_count) )
        capture.release()
    return img_count


def video_gen(root_path, save_path):
    '''
    main function for generating videos
    '''
    file_count = 0
    subjects = os.listdir(root_path)
    for subject in subjects:

        img_path = os.path.join(save_path, subject)

        img_path_0 = img_path + "/left/"
        img_path_1 = img_path + "/front/"
        img_path_2 = img_path + "/right/"

        data_path = os.path.join(root_path, subject)
        files = os.listdir(data_path)
        print("\nProcess with subject: ", data_path)

        video_file_0 = filter(lambda x: x.endswith('0.avi'), files)
        video_file_1 = filter(lambda x: x.endswith('1.avi'), files)
        video_file_2 = filter(lambda x: x.endswith('2.avi'), files)

        video_data_0 = video2frame(video_file_0, data_path, img_path_0)
        # video_data_1 = video2frame(video_file_1, data_path, img_path_1)
        # video_data_2 = video2frame(video_file_2, data_path, img_path_2)

        file_count += 1

    return file_count

root_path = "/media/vpromise/Inter/orthopedic/all_data"
save_path = "/media/vpromise/Inter/orthopedic/2d_data/img_all"

video_gen(root_path, save_path)
