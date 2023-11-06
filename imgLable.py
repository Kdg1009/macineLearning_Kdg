import os
from blender import blender
from getPoint import getPoints
from imgBlend import blend

root_dir='/project/python_project/carDetect/project/'
# Load horizental/vertical line img
confirm = 'n'
colors=['red','purple','black']
redBln=blender(colors[0])
purpleBln=blender(colors[1])
blackBln=blender(colors[2])
blenders=[None,redBln,purpleBln,blackBln]

for dir in os.listdir(root_dir):
    if os.path.isfile(root_dir+dir):
        continue
    for subDir in os.listdir(root_dir+dir):
        labledCheck=list(subDir.split('.'))[-1]
        if labledCheck == 'labled':
            continue
        rename_valid='n'
        for img_file in os.listdir(root_dir+dir+'/'+subDir): 
            fileInfo=list(img_file.split('.'))
            file_extens=fileInfo[-1]
            num=fileInfo[0]
            labelCheck=fileInfo[1]
            if labelCheck[0]=='[':
                continue
            print(dir+'/'+subDir+'/'+img_file)
            #gen line img
            blend(blenders,root_dir+dir+'/'+subDir+'/'+img_file)
            #get points
            points=getPoints()
            #rename file
            os.rename(root_dir+dir+'/'+subDir+'/'+img_file,root_dir+dir+'/'+subDir+'/'+num+'.'+str(points)+'.'+file_extens)
