from __future__ import print_function
import io
import os
import sys
import random
import cv2
import pandas as pd
from PIL import Image
import csv

base = '/media/yyk/My Passport/DL/AffectNet_Database/test/test_imgs/'
done = '/media/yyk/My Passport/DL/AffectNet_Database/test/done_imgs/'
csv_file = '/media/yyk/My Passport/DL/AffectNet_Database/test/test.csv'

fname = []
face_x = []
face_y = []
face_width = []
face_height = []
expression = []
new_val_txt = open('/media/yyk/My Passport/DL/AffectNet_Database/test/val.txt','w')
with open('/media/yyk/My Passport/DL/AffectNet_Database/test/test.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        '''
        print(row['subDirectory_filePath'])
        print(row['face_x'][0:])
        print(row['face_y'][0:])
        print(row['face_width'][0:])
        print(row['face_height'][0:])
        print(row['expression'][0:])
        '''
        fname = row['subDirectory_filePath']
        x = int(row['face_x'][0:])
        y = int(row['face_y'][0:])
        width = int(row['face_width'][0:])
        height = int(row['face_height'][0:])
        expression = row['expression'][0:]
        floder_dir = fname.split('/')[0]
        img = fname.split('/')[1]
        image = cv2.imread(os.path.join(base,fname))
        new_val_txt.write(fname)
        new_val_txt.write(' ')
        new_val_txt.write(expression)
        new_val_txt.write('\n')
        imgROI = image[x:x + width, y:y + height]
        imgROI = cv2.resize(imgROI, (224, 224), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
        if not os.path.isdir('./test/done_imgs/' + floder_dir):
            os.mkdir('./test/done_imgs/' + floder_dir)
        cv2.imwrite('/media/yyk/My Passport/DL/AffectNet_Database/test/done_imgs/' + floder_dir + '/' + img, gray)
        cv2.waitKey(0)