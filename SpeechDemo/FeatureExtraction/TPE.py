
import time
import traceback
import hyperopt
from PIL import Image
from hyperopt import STATUS_OK
import dlib
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from skimage.color import label2rgb
from skimage import measure, filters


# face
# def para(Mouth_right,Mouth_left,ROIpath,Gaborpath,b,mouth_centroid_x,mouth_centroid_y):
#
def backgroundColor(image, mouth_centroid_y, mouth_centroid_x):
    bgc = 0
    # img = Image.open(image)
    size = image.shape
    # print(mouth_centroid_x,mouth_centroid_y)
    pixel1 = int(image[mouth_centroid_y, mouth_centroid_x])
    pixel2 = int(image[mouth_centroid_y - 1, mouth_centroid_x])
    pixel3 = int(image[mouth_centroid_y - 1, mouth_centroid_x - 1])
    pixel4 = int(image[mouth_centroid_y, mouth_centroid_x - 1])
    Average_center = (pixel1 + pixel2 + pixel3 + pixel4) / 4
    left = int(image[0, 0])
    leftbutton = int(image[size[0] - 1, 0])
    right = int(image[0, size[1] - 1])
    rightbutton = int(image[size[0] - 1, size[1] - 1])
    Average_angle = (left + leftbutton + right + rightbutton)/4
    # print(pixel4, Average_angle)
    if Average_angle < 20:
        return bgc
    elif Average_angle > 253:
        bgc = 1
        return bgc
    elif Average_angle < Average_center:
        return bgc
    else:
        bgc = 1
        return bgc


def HGabor(input):

    Hkernel_size0 = input["Hkernel_size"]
    Hwavelength0 = input["Hwavelength"]
    Hsig0 = input["Hsig"]
    gmH= input["Hgamma"]
    # print(Hkernel_size0,Hwavelength0,Hsig0,gmH)
    try:
        imgGray = cv2.cvtColor(ROI_mouth, cv2.COLOR_BGR2GRAY)  # Change color picture into gray picture
        imgGray_f = np.array(imgGray, dtype=np.float64)  # Change data type of picture
        imgGray_f /= 255.

        orientationH = 90  # orientation of normal direction
        kernel_sizeH = int(Hkernel_size0)
        wavelengthH = int(Hwavelength0)
        sigH = int(Hsig0)


        ps = 0.0
        thH = orientationH * np.pi / 180
        # th=0.14
        kernelH = cv2.getGaborKernel((kernel_sizeH, kernel_sizeH), sigH, thH, wavelengthH, gmH, ps)
        # print(kernelH)
        destH = cv2.filter2D(imgGray_f, cv2.CV_32F, kernelH)  # CV_32F
        # print(destH)
        # if
        # cv2.imshow("destH",destH)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        Gabor_Path = '/Users/lexie/SpeechProject/TPEH.jpg'
        cv2.imwrite(Gabor_Path, np.power(destH, 2))

        # fig, ax = plt.subplots(figsize=(10, 6))
        image = cv2.imread(Gabor_Path, 0)
        # image=GaborP
        # print("Image shape",image.shape)
        thresh = filters.threshold_yen(image)  # high thresh
        # thresh = filters.threshold_otsu(image)  #Determination of optimal Segmentation threshold by otsu algorithm
        bwimg = (image >= (thresh))  # Segmenting with threshold to generate binary image
        # print(1)
        bgc = backgroundColor(image, mouth_centroid_y, mouth_centroid_x)
        # print("bgc", bgc)
        labels, num = measure.label(bwimg, return_num=True, background=1)  # Labeled connected region
        image_label_overlay = label2rgb(labels, image=image)




        if num==1:
            # print("num:",num)
                # show picture2

                x1 = mouth_centroid_x
                y1 = mouth_centroid_y
                    # print('x y',x1,y1)

                minw = 1000
                minh = 1000
                for region in measure.regionprops(labels, intensity_image=image, coordinates='rc'):
                        # print(region)
                        minr, minc, maxr, maxc = region.bbox

                            # print('area',region.area)
                            # print(region.centroid)
                        x = region.centroid[1]
                        y = region.centroid[0]

                        w = abs(x1 - x)
                        h = abs(y - y1)
                        # wah = w + h
                            # print 'w,h',w,h

                        if w <= minw and h <=minh:
                            minw=w
                            minh=h
                            min_maxc = maxc
                            min_maxr = maxr
                            min_minc = minc
                            min_minr = minr


                    # print('rx,ry', min_x, min_y)


                width = min_maxc - min_minc
                height = min_maxr-min_minr
                err = abs(widthG - width) + abs(heightG - height)
                f = err

                return {'loss': f, 'status': STATUS_OK}
        else:
                f=1000
                return {'loss': f, 'status': STATUS_OK}
    except Exception as e:
        f = 1000
        return {'loss': f, 'status': STATUS_OK}

#
def TPE(TPEPath,b,shotname,picturepathF,mouth_centroid_xF, mouth_centroid_yF,ROI_mouthF,widthGF,heightGF):

    global picturepath
    picturepath = picturepathF
    print(picturepath)
    global mouth_centroid_x
    mouth_centroid_x = int(mouth_centroid_xF)
    global mouth_centroid_y
    mouth_centroid_y = int(mouth_centroid_yF)
    global ROI_mouth
    ROI_mouth = ROI_mouthF
    global widthG
    widthG = widthGF
    global heightG
    heightG = heightGF


    # print(mouth_centroid_x,mouth_centroid_y,widthG)

    # New Add
    # with open('/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/TPE.csv', encoding='utf-8') as f:
    #     reader = csv.reader(f)
    #     header = next(reader)
    #     # print(header)
    #     for row in reader:
    #         print(row)
    #
    # # New Add
    # Hkernel_size_min= row[0]
    # Hwavelength_min= row[1]
    # Hsig_min= row[2]
    # Hgamma_min= row[3]
    #
    # print("Hkernel_size",Hkernel_size_min)
    with open('/Users/lexie/PycharmProjects/SpeechDemo/TPE.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # print(header)
        for row in reader:
            print(row)
    HKernel_size_min = int(row[0])
    HWavelength_min = int(row[1])
    HSig_min = int(row[2])
    Hgamma_min = float(row[3])
    print("para=", HKernel_size_min, HWavelength_min, HSig_min, Hgamma_min)

    search_spaceH = {
        "Hkernel_size": hyperopt.hp.quniform('Hkernel_size', 14, 17, 1),
        "Hwavelength": hyperopt.hp.quniform('Hwavelength', 14, 17, 1),
        "Hsig": hyperopt.hp.quniform('Hsig', 3, 7, 1),
        "Hgamma": hyperopt.hp.quniform('Hgamma', 0.1, 1.0,0.1)
    }

    # set value
    search_spaceH['Hkernel_size'] = hyperopt.hp.quniform('Hkernel_size', HKernel_size_min, 40, 1)
    search_spaceH['Hwavelength'] = hyperopt.hp.quniform('Hwavelength', HWavelength_min, 40, 1)
    search_spaceH['Hsig'] = hyperopt.hp.quniform('Hsig', HSig_min, 7, 1)
    search_spaceH['Hgamma'] = hyperopt.hp.quniform('Hgamma', Hgamma_min, 1.0, 0.1)
    # print("make sure para=", search_spaceH['Hkernel_size'])
    # print("make sure para=", search_spaceH['Hwavelength'])
    # print("make sure para=", search_spaceH['Hsig'])
    # print("make sure para=", search_spaceH['Hgamma'])

    while True:
        try:
            trials = hyperopt.Trials()
            Hbest = hyperopt.fmin(
                        fn=HGabor,
                        space=search_spaceH,
                        algo=hyperopt.tpe.suggest,
                        max_evals=150,
                        trials=trials
                    )


            trial_loss = np.asarray(trials.losses(), dtype=float)
            best_loss = min(trial_loss)
            print('best loss: ', best_loss)
            if best_loss<=34:
                break
        except Exception as e:

            if "writerow() takes exactly one argument (2 given)" in traceback.format_exc():
                print("Got errors! 2nd Round***********")
            continue
        break



    HGamma=Hbest.get("Hgamma")
    HKernelSize = int(Hbest.get("Hkernel_size"))
    HSig = int(Hbest.get("Hsig"))
    HWavelength= int(Hbest.get("Hwavelength"))


    print("HGamma, HKernelSize, HSig, HWavelength",HGamma, HKernelSize, HSig, HWavelength)

    # New Add

    if not os.path.exists(TPEPath):
        os.mkdir(os.path.join(TPEPath))
    # else:
    #        print 'file exist'
    cur_dir = os.path.join(TPEPath, shotname)
    if not os.path.exists(cur_dir):
        os.mkdir(cur_dir)

    header = ['HGamma', 'HKernelSize', 'HSig', 'HWavelength']
    data = [{'HGamma': HGamma, 'HKernelSize': HKernelSize, 'HSig': HSig, 'HWavelength': HWavelength}]
    value=[HGamma, HKernelSize, HSig, HWavelength]
    cur_dir = cur_dir + '/'  # 保存路径
    # print(path)
    if not os.path.exists(cur_dir):
        os.mkdir(cur_dir)
    TPEPath = cur_dir + str('%02d' % b) + '.csv'

    # 可以write in column
    # with open(TPEPath, 'w', encoding='utf-8', newline='') as f:
    #     writer = csv.DictWriter(f, header)
    #     writer.writeheader()
    #     writer.writerows(data)
    # print("write TPE.csv complete")

    with open(TPEPath, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(0, len(header)):
            writer.writerow([header[i], value[i]])
    # New Add

    return HGamma, HKernelSize, HSig, HWavelength
    # how to determine the background color
