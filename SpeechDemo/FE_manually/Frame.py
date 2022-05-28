#-*- coding: utf-8 -*-
import glob
import cv2
import os
import ROI

def Frame(detector,predictor,VideoPath,PicturePath,Mouth_path,GaborPath,SheetPath,FeaturesPath):
 if not os.path.exists(PicturePath):
        os.mkdir(os.path.join(PicturePath))

 for video in glob.glob(VideoPath): # path of videos
    (filepath, tempfilename) = os.path.split(video)
    (shotname, extension) = os.path.splitext(tempfilename)
    folder_name = shotname
    Path = os.path.join(PicturePath, folder_name)
    if not os.path.exists(Path):
        os.mkdir(Path)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # size=(960,544)
    # print 'size',size

    i = 0
    while (cap.isOpened()):  # cv2.VideoCapture.isOpened()
        i = i + 1
        ret, frame = cap.read()  # cv2.VideoCapture.read()　
        if ret == True:
            path = Path +'/'
            picturepath = path+ str('%02d' % i) + '.jpg'
            # print picturepath
            cv2.imwrite(picturepath, frame)
            ROI.rect1(detector,predictor,i,shotname,picturepath,Mouth_path,GaborPath,SheetPath,FeaturesPath)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break



    cap.release()