import os
import numpy as np
from PIL import Image

name_dict = {
'fyc': 0, 'hy': 1, 'ljg': 2, 'lqf': 3, 'lsl': 4,
'ml': 5, 'nhz': 6, 'rj': 7, 'syj': 8, 'wq': 9,
'wyc': 10, 'xch': 11, 'xxj': 12, 'yjf': 13, 'zdx': 14,
'zjg': 15, 'zyf': 16
}

path = '/home/liuz/detectron2/'

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

def create_testset(angle):
    x_test = []
    y_test = []
    for name in name_dict.keys():
        single_video = []
        full_path = path + angle + '_bbox+mask/test/' + name + '/00_3/'
        label = name_dict[name]
        
        for img_name in findAllFile(full_path):
            img = Image.open(img_name)
            img_array = np.asarray(img)    
            #print(img_array.shape) #(299,299,3)
            single_video.append(img_array)
        
        x_test.append(single_video)
        y_test.append(label)
        
    x_test = np.array(x_test) 
    y_test = np.array(y_test) 
    #print(x_test.shape) #(17, 50, 299, 299, 3)
    #print(y_test.shape) #(17,)
    np.save('gait_dataset/x_test_' + angle + '.npy',x_test)
    np.save('gait_dataset/y_test_' + angle + '.npy',y_test)
    
def create_validset(angle):
    x_test = []
    y_test = []
    for name in name_dict.keys():
        single_video = []
        full_path = path + angle + '_bbox+mask/val/' + name + '/00_4/'
        label = name_dict[name]
        
        for img_name in findAllFile(full_path):
            img = Image.open(img_name)
            img_array = np.asarray(img)    
            single_video.append(img_array)
        
        x_test.append(single_video)
        y_test.append(label)
        
    x_test = np.array(x_test) 
    y_test = np.array(y_test) 
    #print(x_test.shape) #(17, 50, 299, 299, 3)
    #print(y_test.shape) #(17,)    
    np.save('gait_dataset/x_valid_' + angle + '.npy',x_test)
    np.save('gait_dataset/y_valid_' + angle + '.npy',y_test)
    
def create_trainset(angle):
    x_test = []
    y_test = []
    
    full_path_list = [['brightness1_','/00_1/'],['brightness1_','/00_2/'],
                      ['brightness2_','/00_1/'],['brightness2_','/00_2/'],
                      ['','/00_1/'],['','/00_2/'],
                      ['hue_','/00_1/'],['hue_','/00_2/'],
                      ['left_right_','/00_1/'],['left_right_','/00_2/']]
    
    for idx in range(0,10):
        for name in name_dict.keys():
            single_video = []
            full_path = path + angle + '_bbox+mask/train/' + full_path_list[idx][0] + name + full_path_list[idx][1]
            if(os.path.exists(full_path)):

                label = name_dict[name]
        
                for img_name in findAllFile(full_path):
                    img = Image.open(img_name)
                    img_array = np.asarray(img)    
                    single_video.append(img_array)
                
                x_test.append(single_video)
                y_test.append(label)
                #print(np.array(x_test).shape) 
      
    x_test = np.array(x_test) 
    y_test = np.array(y_test) 
    #print(x_test.shape) #(165, 50, 299, 299, 3)
    #print(y_test.shape) #(165,)
    np.save('gait_dataset/x_train_' + angle + '.npy',x_test)
    np.save('gait_dataset/y_train_' + angle + '.npy',y_test)
    
if __name__ == "__main__":
    
    create_testset('0')
    create_validset('0')
    create_trainset('0')
    

