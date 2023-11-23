import numpy as np
import cnnLayers_v2 as cn

# 1. get feature map using cnn output.shape=(gridH,gridW,#Bbox*5(x,y,w,h,boxconfidence)*C(#conditional class probability)=>img devided by grid cell
# 3. get classification L+localization L+confidence L
# 4. backpropagation
# 5. use NMS for testing model
# test process: resize img(get grid)=>run cnn model(get feature map)=>NMS

