import os
import numpy as np

DATA_DIR = '/media/xn/AA1A74301A73F821/wbw/CV/FPN_TensorFlow-master/data/pascal/'

anno_files = [file for file in os.listdir(DATA_DIR+'Annotations')]
pic_files = [file for file in os.listdir(DATA_DIR+'JPEGImages')]
print(np.shape(anno_files), np.shape(pic_files))

filter_pics = []
for anno_file in anno_files:
    filter_pic = anno_file.replace('.xml', '.jpg')
    filter_pics.append(filter_pic)
print(np.shape(filter_pics))

filter_num = 0
for file in pic_files:
    if not file in filter_pics:
        filter_num += 1
        os.remove(DATA_DIR+'JPEGImages/'+file)
        print(filter_num, file)