import glob
import cv2
import os
import ROI
import Gabor
import Features
import Frame
import dlib
import time
def ReadVideoPath():
    # a=VideoPath.lstrip()
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

Video_path=ReadVideoPath()


# Video_path='/Users/lexie/PycharmProjects/SpeechDemo/shang4.mp4'
print("Videopath",Video_path)
PicturePath = '/Users/lexie/SpeechProject/Qs6/APicture/'# path to store pictures
MouthPath = '/Users/lexie/SpeechProject/Qs6/Amouth/'  # path to store mouth
GaborPath = '/Users/lexie/SpeechProject/Qs6/AGabor'#path to store Gabor features
SheetPath = '/Users/lexie/SpeechProject/Qs6/ASheet/' # path to storSheetPath
FeaturesPath = '/Users/lexie/SpeechProject/Qs6/AFeatures/'  # path to store sheets
Frame.Frame(detector,predictor,Video_path,PicturePath,MouthPath,GaborPath,SheetPath,FeaturesPath)
b=time.time()
print(b-a)


