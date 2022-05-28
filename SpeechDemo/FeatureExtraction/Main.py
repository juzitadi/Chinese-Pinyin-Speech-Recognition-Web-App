import glob
import cv2
import os
import WordVideo
import dlib
import time
import ROI
import TPE
import Gabor
import Features
import WebApp
# from os import startfile

def ReadVideoPath():
    with open("test.txt", "r") as f:
        VideoPath=f.readline()
        print("VideoPath",VideoPath)

    file = open( "test.txt", "w+" )     # 文件如果不存在就创建
    file.truncate()
    file.close()
    print("clean")

    return VideoPath

a="asaaaaa"
print("received",a)


a=time.time()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/shape_predictor_68_face_landmarks.dat")
VideoPath=ReadVideoPath()
print("Videopath",VideoPath)
Frame = '/Users/lexie/SpeechProject/Qs6/APicture/'# path to store pictures
MouthPath = '/Users/lexie/SpeechProject/Qs6/Amouth/'  # path to store mouth
GaborPath = '/Users/lexie/SpeechProject/Qs6/AGabor'#path to store Gabor features
SheetPath = '/Users/lexie/SpeechProject/Qs6/ASheet/' # path to storSheetPath
FeaturesPath = '/Users/lexie/SpeechProject/Qs6/AFeatures/'  # path to store sheets
TPEPath = '/Users/lexie/SpeechProject/Qs6/ATPE/'

WordVideo.Frame(detector,predictor,VideoPath,Frame,MouthPath,GaborPath,SheetPath,FeaturesPath,TPEPath)

b = time.time()
print("Time = ",b-a)