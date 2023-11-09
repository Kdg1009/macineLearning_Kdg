import numpy as np
import cv2

colors={'red':(0,0,255),'purple':(255,0,255),'black':(0,0,0)}

def blender(color):
    blender=np.ones((427,640,3),dtype=np.uint8)
    for i in range(0,640,10):
        cv2.line(blender,[i,0],[i,427],colors.get(color))
        if i % 50 == 0:
            cv2.putText(blender,str(i),[i-5,20],cv2.FONT_HERSHEY_COMPLEX,0.4,colors.get('red'),1)
            cv2.putText(blender,str(i),[i-5,413],cv2.FONT_HERSHEY_COMPLEX,0.4,colors.get('red'),1)
            cv2.line(blender,[i,0],[i,5],(255,255,255))
            cv2.line(blender,[i,427],[i,422],(255,255,255))
    for i in range(0,427,10):
        cv2.line(blender,[0,i],[640,i],colors.get(color))
        if i % 50 == 0:
            cv2.putText(blender,str(i),[0,i+2],cv2.FONT_HERSHEY_COMPLEX,0.4,colors.get('red'),1)
            cv2.putText(blender,str(i),[620,i+2],cv2.FONT_HERSHEY_COMPLEX,0.4,colors.get('red'),1)
            cv2.line(blender,[0,i],[5,i],(255,255,255))
            cv2.line(blender,[640,i],[635,i],(255,255,255))
    return blender