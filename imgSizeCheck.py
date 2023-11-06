import cv2
import os

path='/project/python_project/carDetect/project/test_set/test_3.labled/1.[[(0,0), (0,0)]].jpg'
img=cv2.imread(path)
print(img.shape)
print(img[2][428][0])
