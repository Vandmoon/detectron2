import numpy as np
import cv2
import os
import csv

img_path = 'img/'
# label文件为用对应颜色绘制的gt图片
lbl_path = 'lbl/'

imgfiles = os.listdir(img_path)
imgfiles.sort()
lblfiles = os.listdir(lbl_path)
lblfiles.sort()

# 类别数量，包括背景
num_classes = 3 
SIZE = 256

# 颜色的种类，用opencv要注意rgb和bgr
label_values = label_values = [[255, 255, 0], [255, 255, 255], [0, 0, 0]]

# 将label变为one hot形式
def one_hot_it(label, label_values):
    semantic_map = []
    for colour in label_values:       
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis = -1)
    return semantic_map

imgdata = []
lbldata = []
for img,lbl in zip(imgfiles, lblfiles):

    image = cv2.imread(img_path + img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (SIZE, SIZE))
    image = np.float32(image) 
    imgdata.append(image)

    lbl_image = cv2.imread(lbl_path + lbl)
    lbl_image = cv2.cvtColor(lbl_image, cv2.COLOR_BGR2RGB) # 和20行的颜色类别对应，需要转为rgb
    lbl_image = cv2.resize(lbl_image, (SIZE, SIZE))
    lbl_image = one_hot_it(lbl_image, label_values) # 遍历颜色label，生成one hot标签
    
    lbldata.append(lbl_image)
    
# train_img.npy的shape为(SIZE, SIZE, 3)，这里的3是图片的通道数
np.save('train_img', imgdata)
# train_lbl.npy的shape为(SIZE, SIZE, 3)，这里的3为类别的数目
np.save('train_lbl', lbldata)

