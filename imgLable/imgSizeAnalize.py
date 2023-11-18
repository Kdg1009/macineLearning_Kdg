from PIL import Image
import os
import cv2
import traceback

root_dir='/project/python_project/carDetect/project/'

img_size={}
for dir in os.listdir(root_dir):
    if os.path.isfile(root_dir+dir):
        continue
    for subDir in os.listdir(root_dir+dir):
        for img_f in os.listdir(root_dir+dir+'/'+subDir):
            try:
                img=Image.open(root_dir+dir+'/'+subDir+'/'+img_f)
                img_s = img.size
                if img_size.get(img_s):
                    num=img_size[img_s]
                    num += 1
                    img_size[img_s]=num
                else:
                    img_size[img_s]=1
            except OSError as ex:
                print('error occured! ')

print(img_size)