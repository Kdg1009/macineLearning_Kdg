from PIL import Image
import os
import cv2

def equal(ob1,ob2):
    i = len(ob1)
    j = len(ob2)

    if not i == j:
        return False
    for time in range(i):
        if not ob1[time] == ob2[time]:
            return False
    return True
                
root_dir='/project/python_project/carDetect/project/'

for dir in os.listdir(root_dir):
    if os.path.isfile(root_dir+dir):
        continue
    elif dir=='__pycache__':
        continue
    for sudDir in os.listdir(root_dir+dir):
        for img_file in os.listdir(root_dir+dir+'/'+sudDir):
            
            try:
                img=Image.open(root_dir+dir+'/'+sudDir+'/'+img_file)
                if not equal(img.size,((640,427))):
                    img = img.convert("RGB")
                    new_img=img.resize((640,427))
                    new_img.save(root_dir+dir+'/'+sudDir+'/'+img_file)
            except Exception:
                print('error occured!')