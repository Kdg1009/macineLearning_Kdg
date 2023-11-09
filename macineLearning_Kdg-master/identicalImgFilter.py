import imgHash_v2
import os
import cv2


root_dir='/project/python_project/carDetect/project/'
for dir in os.listdir(root_dir):
    if os.path.isfile(root_dir+dir):
        continue
    elif dir=='__pycache__':
        continue
    for sudDir in os.listdir(root_dir+dir):
        for img_file in os.listdir(root_dir+dir+'/'+sudDir):
            file_prospect=img_file.split('.')
            if len(file_prospect)>3:
                continue
            print(sudDir+'/'+img_file)
            img=cv2.imread(root_dir+dir+'/'+sudDir+'/'+img_file)
            hashVal=imgHash_v2.hash(img)
            os.rename(root_dir+dir+'/'+sudDir+'/'+img_file,root_dir+dir+'/'+sudDir+'/'+file_prospect[0]+'.'+file_prospect[1]+'.'+str(hashVal)+'.'+file_prospect[2])