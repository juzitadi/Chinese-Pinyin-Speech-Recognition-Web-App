
import cv2
import numpy as np
import pylab as pl
from PIL import Image
from PIL import ImageFilter
import os
import glob
import Features

def Gabor_h(HGamma, HKernelSize, HSig, HWavelength,i,ROIpath,shotname,GaborPath):
    print(shotname)
    #cur_dir2 = 'D:/codepython3/Gabor/' # path to store Gabor feature
    if not os.path.exists(GaborPath):
        os.mkdir(os.path.join(GaborPath))
    print(shotname)
    Gaborpath = os.path.join(GaborPath, shotname)
    if not os.path.exists(Gaborpath):
        os.mkdir(Gaborpath)

    img = cv2.imread(ROIpath, 1)  # Loading color picture

    cv2.imshow("1",img)
    cv2.waitKey(200)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Change color picture into gray picture
    imgGray_f = np.array(imgGray, dtype=np.float32)  # Change data type of picture
    imgGray_f /= 255.

    # Parameters of horizontal filter
    orientationH = 90  # orientation of normal direction
    wavelengthH = HWavelength
    kernel_sizeH = HKernelSize
    sigH = HSig
    gmH = HGamma
    print(1)
    ps = 0.0
    thH = orientationH * np.pi / 180
    kernelH = cv2.getGaborKernel((kernel_sizeH, kernel_sizeH), sigH, thH, wavelengthH, gmH, ps)
    destH = cv2.filter2D(imgGray_f, cv2.CV_32F, kernelH)  # CV_32F
    Gaborpath = Gaborpath + '/'   # 保存路径
    print("Gaborpath=",Gaborpath)
    if not os.path.exists(Gaborpath):
        os.mkdir(Gaborpath)
    Gabor_Path = Gaborpath  + str('%02d' % i) + '.jpg'
    print("Gabor=",Gabor_Path)
    cv2.imwrite(Gabor_Path, np.power(destH, 2))
    return Gabor_Path

    #return i,shotname,Gabor_Path