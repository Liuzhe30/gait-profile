import os
import json

name_dict = {'fyc': 0, 'hy': 1, 'ljg': 2, 'lqf': 3, 'lsl': 4,
             'ml': 5, 'nhz': 6, 'rj': 7, 'syj': 8, 'wq': 9,
             'wyc': 10, 'xch': 11, 'xxj': 12, 'yjf': 13, 'zdx': 14,
             'zjg': 15, 'zyf': 16}
path = '/home/liuz/detectron2/gait_dataset/'
partition = {}
labels = {}

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f
            
def create_validjson():
    
    partition['validation'] = []
    full_path = path + 'valid_data/'
    for name in name_dict:
        partition['validation'].append('valid_data/' + name + '_valid')
        labels['valid_data/' + name + '_valid'] = name_dict[name]
    #print(partition)
    #print(labels)

def create_trainjson():
    
    partition['train'] = []
    full_path = path + 'train_data/'
    for video_name in findAllFile(full_path):
        partition['train'].append('train_data/' + video_name.split('.')[0])
        labels['train_data/' + video_name.split('.')[0]] = name_dict[video_name.split('.')[0].split('_')[0]]
    #print(partition)
    #print(labels)

def create_json():
    create_validjson()
    create_trainjson() 
    return partition, labels
    
if __name__ == "__main__":
    
    create_validjson()
    create_trainjson()
    