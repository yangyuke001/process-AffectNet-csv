from __future__ import print_function
import io
import os
import sys
import random
import cv2
import pandas as pd
from PIL import Image
import csv

base = '/media/yyk/My Passport/datasets/face_emotion/affectNet/Manually_Annotated/Manually_Annotated_Images'
done = '/media/yyk/My Passport/datasets/face_emotion/affectNet/Manually_train_croped/'
csv_file = '/media/yyk/My Passport/datasets/face_emotion/affectNet/Manually_Annotated_file_lists/training.csv'
new_val_txt = open('/media/yyk/My Passport/datasets/face_emotion/affectNet/Manually_Annotated_file_lists/train.txt','w')

fname = []
face_x = []
face_y = []
face_width = []
face_height = []
expression = []
num = 0
with open(csv_file,'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        num += 1
        fname = row['subDirectory_filePath']
        x = int(row['face_x'][0:])
        y = int(row['face_y'][0:])
        width = int(row['face_width'][0:])
        height = int(row['face_height'][0:])
        expression = int(row['expression'][0:])
        floder_dir = fname.split('/')[0]
        img = fname.split('/')[1]
        image = cv2.imread(os.path.join(base,fname))

        #write name & expression to new txt
        if expression < 7:
            new_val_txt.write(fname)
            new_val_txt.write(' ')
            new_val_txt.write(str(expression))
            new_val_txt.write('\n')
            
        #process img
        try:
            imgROI = image[x:x + width, y:y + height]
        except:
            pass
        imgROI = cv2.resize(imgROI, (224, 224), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
        if not os.path.isdir('./Manually_train_croped/' + floder_dir):
            os.mkdir('./Manually_train_croped/' + floder_dir)
        cv2.imwrite(done + floder_dir + '/' + img, gray)
        print(fname)
        cv2.waitKey(0)
 
    print(num)
