#-*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as pl
from PIL import Image
from PIL import ImageFilter
import os
import glob
import Features
import csv

def Gabor_h(mouth_centroid_x, mouth_centroid_y,i,ROIpath,shotname,GaborPath,SheetPath,FeaturesPath):

    #cur_dir2 = 'D:/codepython3/Gabor/'#path to store Gabor features
    if not os.path.exists(GaborPath):
        os.mkdir(os.path.join(GaborPath))
    Gaborpath = os.path.join(GaborPath, shotname)
    if not os.path.exists(Gaborpath):
        os.mkdir(Gaborpath)

    img=cv2.imread(ROIpath)                          # Loading color picture
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Change color picture into gray picture
    imgGray_f = np.array(imgGray,dtype=np.float64)   # Change data type of picture
    imgGray_f /=255.

    orentation = 90
    ps = 0.0
    th = orentation * np.pi / 180
    with open('/Users/lexie/PycharmProjects/SpeechDemo/TPE.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # print(header)
        for row in reader:
            print(row)
    kernel_size = int(row[0])
    wavelength = int(row[1])
    sig = int(row[2])
    gm = float(row[3])
    print("para=", kernel_size,wavelength,sig,gm)

    #tune value!!!!!!!!!!!!!!!
    # wavelenth = 16
    # kernel_size = 12    #12
    # sig =5                           #bandwidth
    # gm = 0.1

        #th=0.14
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sig, th,wavelength,gm,ps)
    kernelimg=kernel /2 + 0.5
    dest = cv2.filter2D(imgGray_f, cv2.CV_32F, kernel)#CV_32F
    Gabor_Path = Gaborpath + '/'+str('%02d' % i) + '.jpg'
    cv2.imwrite(Gabor_Path, np.power(dest, 2))

    Features.Features(mouth_centroid_x, mouth_centroid_y,i,shotname,Gabor_Path,SheetPath,FeaturesPath)
    #return i,shotname,Gabor_Path