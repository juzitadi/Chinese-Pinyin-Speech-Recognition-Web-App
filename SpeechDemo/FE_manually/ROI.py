#-*- coding: utf-8 -*-




import numpy as np
import cv2
import os
from PIL import Image
import scipy.ndimage as ndi
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import io,measure,color,data,filters
import Gabor
import dlib
#face
def rect1(detector,predictor,i,shotname,picturepath,MouthPath,GaborPath,SheetPath,FeaturesPath):


 # detector = dlib.get_frontal_face_detector()
 #
 # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #已训练好的模型，存储在代码文件夹内，直接调用

 img = cv2.imread(picturepath)
    # show picture1
 # win = dlib.image_window()
 # win.clear_overlay()
 # win.set_image(img)


 dets = detector(img, 1)

 # create a file PATH to store mouth picture

 if not os.path.exists(MouthPath):
    os.mkdir(os.path.join(MouthPath))
 cur_dir =os.path.join(MouthPath, shotname)
 if not os.path.exists(cur_dir):
     os.mkdir(os.path.join(cur_dir))
 # else:
 #     print 'file exist'


    #face
 for k, d in enumerate(dets):
     shape = predictor(img, d)

     t1 = shape.part(48).x - 10
     t2 = shape.part(54).x + 10
     t3 = shape.part(50).y - 5
     t4 = shape.part(58).y + 5



     # print("ROI image=",t1,t2,t3,t4)

     mouth_centroid_x = (shape.part(48).x-t1) + abs(shape.part(54).x - shape.part(48).x) / 2
     mouth_centroid_y = (shape.part(51).y- t4) + abs(shape.part(62).y - shape.part(51).y) + abs(shape.part(66).y - shape.part(62).y) / 2
     ROI_mouth = img[t3:t4, t1:t2]
     ROIpath = cur_dir + '/' + str('%02d' % i) + '.jpg'
     cv2.imwrite(ROIpath, ROI_mouth)
     Gabor.Gabor_h(mouth_centroid_x, mouth_centroid_y, i, ROIpath, shotname,GaborPath,SheetPath,FeaturesPath)

