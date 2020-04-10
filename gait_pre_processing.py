import os
import numpy as np

name_dict = {
'fyc': 0, 'hy': 1, 'ljg': 2, 'lqf': 3, 'lsl': 4,
'ml': 5, 'nhz': 6, 'rj': 7, 'syj': 8, 'wq': 9,
'wyc': 10, 'xch': 11, 'xxj': 12, 'yjf': 13, 'zdx': 14,
'zjg': 15, 'zyf': 16
}

path = '/home/liuz/detectron2/'
path_0 = path + '0_bbox+mask/'
path_45 = path + '45_bbox+mask/'

def create_testset():
    x_test = []
    y_test = []
    for name in name_dict.keys():
        single_video = []
        full_path = path_0 + 'test/' + name + '/00_3/'
        label = name_dict[name]
        


