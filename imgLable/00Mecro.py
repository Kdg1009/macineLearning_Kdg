import os

root_dir='/project/python_project/carDetect/project'
fileDir=[None]

for subDir in os.listdir(root_dir):
    if os.path.isfile(root_dir+'/'+subDir):
        continue
    labled=list(subDir.split('.'))[-1]
    if not labled == 'labled':
        fileDir.append(subDir)
fileSubDir=input('test 1/train 2: ')
fileSubDir = int(fileSubDir)
root_dir = root_dir +'/'+fileDir[fileSubDir]
while not len(fileDir) == 1:
    fileDir.pop(-1)

for subDir in os.listdir(root_dir):
    if not list(subDir.split('.'))[-1] == 'labled':
        fileDir.append(subDir)
print(fileDir)
fileSubDir=input(': ')
fileSubDir=int(fileSubDir)
root_dir = root_dir+'/'+fileDir[fileSubDir]

for img in os.listdir(root_dir):
    fileInfo=list(img.split('.'))
    os.rename(root_dir+'/'+img,root_dir+'/'+fileInfo[0]+'.[[(0,0), (0,0)]].'+fileInfo[-1])