import os
import csv
import pandas as pd
import numpy as np

# marker information
# marker_order: 00 脖子 01 胸前
marker_order = ['Marker_filtered_JN-X',     'Marker_filtered_JN-Y',     'Marker_filtered_JN-Z',      # 00 脖子
                'Marker_filtered_XIPH-X',   'Marker_filtered_XIPH-Y',   'Marker_filtered_XIPH-Z']    # 01 胸前

# marker_left: 00 左髋 01 左膝 02 左踝
marker_left = [ 'Marker_filtered_LASIS-X',  'Marker_filtered_LASIS-Y',  'Marker_filtered_LASIS-Z',   # 00 左髋
                'Marker_filtered_LLEK-X',   'Marker_filtered_LLEK-Y',   'Marker_filtered_LLEK-Z',    # 01 左膝
                'Marker_filtered_LLM-X',    'Marker_filtered_LLM-Y',    'Marker_filtered_LLM-Z']     # 02 左踝

# marker_right: 00 右髋 01 右膝 02 右踝
marker_right = ['Marker_filtered_RASIS-X',  'Marker_filtered_RASIS-Y',  'Marker_filtered_RASIS-Z',   # 00 右髋
                'Marker_filtered_RLEK-X',   'Marker_filtered_RLEK-Y',   'Marker_filtered_RLEK-Z',    # 01 右膝
                'Marker_filtered_RLM-X',    'Marker_filtered_RLM-Y',    'Marker_filtered_RLM-Z']     # 02 右踝

# marker_left_all: 00 脖子 01 胸前 02 左髋 03 左膝 04 左踝
marker_left_all = [ 'Marker_filtered_JN-X',     'Marker_filtered_JN-Y',     'Marker_filtered_JN-Z',      # 00 脖子
                    'Marker_filtered_XIPH-X',   'Marker_filtered_XIPH-Y',   'Marker_filtered_XIPH-Z',    # 01 胸前
                    'Marker_filtered_LASIS-X',  'Marker_filtered_LASIS-Y',  'Marker_filtered_LASIS-Z',   # 02 左髋
                    'Marker_filtered_LLEK-X',   'Marker_filtered_LLEK-Y',   'Marker_filtered_LLEK-Z',    # 03 左膝
                    'Marker_filtered_LLM-X',    'Marker_filtered_LLM-Y',    'Marker_filtered_LLM-Z']     # 04 左踝

# marker_right_all: 00 脖子 01 胸前 02 右髋 03 右膝 04 右踝
marker_right_all = [ 'Marker_filtered_JN-X',     'Marker_filtered_JN-Y',     'Marker_filtered_JN-Z',      # 00 脖子
                     'Marker_filtered_XIPH-X',   'Marker_filtered_XIPH-Y',   'Marker_filtered_XIPH-Z',    # 01 胸前
                     'Marker_filtered_RASIS-X',  'Marker_filtered_RASIS-Y',  'Marker_filtered_RASIS-Z',   # 02 右髋
                     'Marker_filtered_RLEK-X',   'Marker_filtered_RLEK-Y',   'Marker_filtered_RLEK-Z',    # 03 右膝
                     'Marker_filtered_RLM-X',    'Marker_filtered_RLM-Y',    'Marker_filtered_RLM-Z']     # 04 右踝

# marker_front_all: 00 脖子 01 胸前 02 左髋 03 左膝 04 左踝 05 右髋 06 右膝 07 右踝
marker_front_all = ['Marker_filtered_JN-X',     'Marker_filtered_JN-Y',     'Marker_filtered_JN-Z',      # 00 脖子
                    'Marker_filtered_XIPH-X',   'Marker_filtered_XIPH-Y',   'Marker_filtered_XIPH-Z',    # 01 胸前
                    'Marker_filtered_LASIS-X',  'Marker_filtered_LASIS-Y',  'Marker_filtered_LASIS-Z',   # 02 左髋
                    'Marker_filtered_LLEK-X',   'Marker_filtered_LLEK-Y',   'Marker_filtered_LLEK-Z',    # 03 左膝
                    'Marker_filtered_LLM-X',    'Marker_filtered_LLM-Y',    'Marker_filtered_LLM-Z',     # 04 左踝
                    'Marker_filtered_RASIS-X',  'Marker_filtered_RASIS-Y',  'Marker_filtered_RASIS-Z',   # 05 右髋
                    'Marker_filtered_RLEK-X',   'Marker_filtered_RLEK-Y',   'Marker_filtered_RLEK-Z',    # 06 右膝
                    'Marker_filtered_RLM-X',    'Marker_filtered_RLM-Y',    'Marker_filtered_RLM-Z']     # 07 右踝

# all 21 markers
marker  =  ['Marker_filtered_JN-X',     'Marker_filtered_JN-Y',     'Marker_filtered_JN-Z',      # 00 脖子
            'Marker_filtered_LASIS-X',  'Marker_filtered_LASIS-Y',  'Marker_filtered_LASIS-Z',   # 01 左髋
            'Marker_filtered_LHEE-X',   'Marker_filtered_LHEE-Y',   'Marker_filtered_LHEE-Z',    # 02 左脚后
            'Marker_filtered_LLEK-X',   'Marker_filtered_LLEK-Y',   'Marker_filtered_LLEK-Z',    # 03 左膝
            'Marker_filtered_LLM-X',    'Marker_filtered_LLM-Y',    'Marker_filtered_LLM-Z',     # 04 左踝
            'Marker_filtered_LLSHA-X',  'Marker_filtered_LLSHA-Y',  'Marker_filtered_LLSHA-Z',   # 05 左小腿
            'Marker_filtered_LLTHI-X',  'Marker_filtered_LLTHI-Y',  'Marker_filtered_LLTHI-Z',   # 06 左大腿
            'Marker_filtered_LMT2-X',   'Marker_filtered_LMT2-Y',   'Marker_filtered_LMT2-Z',    # 07 左趾2
            'Marker_filtered_LMT5-X',   'Marker_filtered_LMT5-Y',   'Marker_filtered_LMT5-Z',    # 08 左趾5
            'Marker_filtered_LPSIS-X',  'Marker_filtered_LPSIS-Y',  'Marker_filtered_LPSIS-Z',   # 09 左腰后
            'Marker_filtered_RASIS-X',  'Marker_filtered_RASIS-Y',  'Marker_filtered_RASIS-Z',   # 10 右髋
            'Marker_filtered_RHEE-X',   'Marker_filtered_RHEE-Y',   'Marker_filtered_RHEE-Z',    # 11 右脚后
            'Marker_filtered_RLEK-X',   'Marker_filtered_RLEK-Y',   'Marker_filtered_RLEK-Z',    # 12 右膝
            'Marker_filtered_RLM-X',    'Marker_filtered_RLM-Y',    'Marker_filtered_RLM-Z',     # 13 右踝
            'Marker_filtered_RLSHA-X',  'Marker_filtered_RLSHA-Y',  'Marker_filtered_RLSHA-Z',   # 14 右小腿
            'Marker_filtered_RLTHI-X',  'Marker_filtered_RLTHI-Y',  'Marker_filtered_RLTHI-Z',   # 15 右大腿
            'Marker_filtered_RMT2-X',   'Marker_filtered_RMT2-Y',   'Marker_filtered_RMT2-Z',    # 16 右趾2
            'Marker_filtered_RMT5-X',   'Marker_filtered_RMT5-Y',   'Marker_filtered_RMT5-Z',    # 17 右趾5
            'Marker_filtered_RPSIS-X',  'Marker_filtered_RPSIS-Y',  'Marker_filtered_RPSIS-Z',   # 18 右腰后
            'Marker_filtered_T10-X',    'Marker_filtered_T10-Y',    'Marker_filtered_T10-Z',     # 19 胸后
            'Marker_filtered_XIPH-X',   'Marker_filtered_XIPH-Y',   'Marker_filtered_XIPH-Z']    # 20 胸前

def csv2np(csv_file, data_path):

    # start and end time, [5s:45s] for 2D train
    start_time = 10
    end_time = 30
    skip_timekey = 2
    
    frequency = 100
    start_timekey = frequency * start_time + 1
    end_timekey = frequency * end_time

    for csv in csv_file:
        csv_full_path = os.path.join(data_path, csv)
        # print(csv_full_path)
        marker_coord = pd.read_csv(csv_full_path, usecols=lambda x: x in marker_right_all)[marker_right_all]
        marker_coord = np.array(marker_coord).reshape(-1,5,3)
        marker_coord = marker_coord[start_timekey:end_timekey:skip_timekey]
    return marker_coord

def csv_gen(root_path, save_path):
    file_count = 0
    subjects = os.listdir(root_path)

    for subject in subjects:

        # load data
        data_path = os.path.join(root_path, subject)
        files = os.listdir(data_path)

        csv_file = filter(lambda x: x.endswith('.csv'), files)
        # c3d_file = filter(lambda x: x.endswith('.c3d'), files)
        csv_data = csv2np(csv_file, data_path)#.reshape(-1,3)

        # save data
        # os.makedirs(os.path.join(save_path, subject))
        csv_save_path = os.path.join(save_path, subject) + "/right_3d.npy"
        # os.remove(csv_save_path)

        print("\nprocess with files:", data_path)
        print("Saving files at:", csv_save_path)
        np.save(csv_save_path, csv_data)

        file_count += 1

    print("\nProcessed with %d files!" % file_count)
    return file_count

root_path = "/media/vpromise/Inter/orthopedic/all_data"
save_path = "/media/vpromise/Inter/orthopedic/3d_data_cons/coor_3d/"

csv_gen(root_path, save_path)