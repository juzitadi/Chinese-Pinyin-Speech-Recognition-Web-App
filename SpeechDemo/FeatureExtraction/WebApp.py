# 0-- Import Libraries

import pandas as pd
import numpy as np
from keras.models import load_model
import streamlit as st
from subprocess import call
import glob
import cv2
import os
import csv
import plotly.express as px
import string
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')


# There are functions used in this project.

def get_data(filename):
    test_data = pd.read_csv(filename)

    return test_data


def write_address(file_path):
    with open("test.txt", "w")as f:
        f.write(file_path)


def display_frames(video_name):
    images = [cv2.imread(file) for file in
              glob.glob("/Users/lexie/SpeechProject/Qs6/APicture/" + video_name + "/*.jpg")]  # frames in each video (1)
    idx = 0
    for _ in range(len(images) - 1):
        cols = st.beta_columns(4)

        if idx < len(images):
            cols[0].image(images[idx], width=550)
        idx += 1

        if idx < len(images):
            cols[1].image(images[idx], width=550)
        idx += 1

        if idx < len(images):
            cols[2].image(images[idx], width=550)
        idx += 1
        if idx < len(images):
            cols[3].image(images[idx], width=550)
            idx = idx + 1
        else:
            break


def display_gabor_extraction(video_name):
    images = [cv2.imread(file) for file in
              glob.glob("/Users/lexie/SpeechProject/Qs6/APicture/" + video_name + "/*.jpg")] # frames in each video (1)
    mouths = [cv2.imread(file) for file in glob.glob("/Users/lexie/SpeechProject/Qs6/Amouth/" + video_name + "/*.jpg")] # frames in each video (2)
    gabors = [cv2.imread(file) for file in glob.glob("/Users/lexie/SpeechProject/Qs6/AGabor/" + video_name + "/*.jpg")] # frames in each video (3)
    features = [cv2.imread(file) for file in
                glob.glob("/Users/lexie/SpeechProject/Qs6/AFeatures/" + video_name + "/*.png")] # frames in each video (4)
    idx = 0
    for _ in range(len(images) - 1):
        cols = st.beta_columns(4)

        if idx < len(images):
            cols[0].image(images[idx], width=350)

        if idx < len(images):
            cols[1].image(mouths[idx], width=350)

        if idx < len(images):
            cols[2].image(gabors[idx], width=350)

        if idx < len(images):
            cols[3].image(features[idx], width=350)
            idx = idx + 1

        else:
            break


def display_visualization(video_name):
    info = os.listdir('/Users/lexie/SpeechProject/Qs6/ATPE/' + video_name) # frames in each video (5)
    info = sorted(info)
    df = pd.DataFrame([], columns=['HGamma', 'HKernelSize', 'HSig', 'HWavelength', 'ID'])
    for i in info:
        data = pd.read_csv('/Users/lexie/SpeechProject/Qs6/ATPE/che1/%s' % i, header=None)
        data.columns = ['name', 'value']
        data = data.set_index(['name'])
        data = data.T
        df = df.append(data)
    df = df.reset_index()
    df['ID'] = np.arange(len(df))
    line_chart_data = df.copy()
    fig1 = px.line(x=line_chart_data["ID"], y=line_chart_data["HGamma"], title="HGamma value in every frame")
    fig2 = px.line(x=line_chart_data["ID"], y=line_chart_data["HKernelSize"], title="HKernelSize value in every frame")
    fig3 = px.line(x=line_chart_data["ID"], y=line_chart_data["HSig"], title="HSig value in every frame")
    fig4 = px.line(x=line_chart_data["ID"], y=line_chart_data["HWavelength"], title="HWavelength value in every frame")

    fig1.update_layout(
        width=300,
        height=300,
    )

    fig2.update_layout(
        width=300,
        height=300
    )
    fig3.update_layout(
        width=300,
        height=300
    )
    fig4.update_layout(
        width=300,
        height=300
    )

    interactive_container = st.beta_container()

    with interactive_container:
        col1, col2 = st.beta_columns(2)
        with col1:
            st.write(fig1)
            st.write(fig2)

        with col2:
            st.write(fig3)
            st.write(fig4)


def feature_extraction_All_Frames():
    call(["python", "/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/Main.py"])
    st.success("Complete features extraction!")


def feature_extraction_Manually():
    call(["python", "/Users/lexie/PycharmProjects/SpeechDemo/FE_manually/Main.py"])
    st.success("Complete features extraction!")


def feature_extraction_First_Frame():
    call(["python", "/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction-1st_Frame/Main.py"])
    st.success("Complete features extraction!")


def main():
    # 1-- Cutomize the app
    st.markdown(

        f"""
    <style>
        
        .reportview-container .main .block-container{{
            max-width: 90%;
            padding-top: 5rem;
            padding-right: 5rem;
            padding-left: 5rem;
            padding-bottom: 5rem;
        }}
        img{{
        	max-width:40%;
        	margin-bottom:40px;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    tabs = ["About", "Try Demo Video", "Upload Video"]
    page1 = st.sidebar.selectbox("Page", tabs)
    # page2=st.sidebar.button("About")

    if page1 == "Upload Video":
        st.title("Try to do prediction by yourself! 🧚‍♀️")
        st.write("Please upload the file here.")
        input = st.file_uploader('')
        if input:
            name = input.name
            st.subheader("Current file use is:" + name)
            address = "/Users/lexie/PycharmProjects/SpeechDemo/" + name
            print(address)

            video_path = address
            video_name = name[0:name.rfind('.')]
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            write_address(video_path)

            if st.checkbox("Start"):

                # if st.button("Go to Step 2"):
                st.subheader("2. Data Visualization and Modification")
                st.write("Please choose the way used to do the Gabor feature extarction.")
                tabs = ["Manually", "First Frame", "All Frames"]
                Gabor_Method = st.selectbox("Gabor Extraction Method", tabs)

                if Gabor_Method == "Manually":
                    st.subheader("All frames would use the parameters you choose to extract the Gabor features.")
                    header = ['HKernelSize', 'HWavelength', 'HSig', 'HGamma']
                    st.header("Please choose the parameter.")
                    col1, col2 = st.beta_columns(2)
                    with col1:
                        HKernelSize = st.number_input(value=12, label="HKernelSize", min_value=10, max_value=17)
                        HWavelength = st.number_input(value=16, label="HWavelength", min_value=10, max_value=17)
                    with col2:
                        HSig = st.number_input(value=3, label="HSig", min_value=1, max_value=7)
                        HGamma = st.number_input(value=0.1, label="HGamma", min_value=0.1, max_value=1.0)

                if Gabor_Method == "First Frame":
                    st.subheader("You choose the range of parameters for the first frame. "
                                 "Then the following will use the same parameters used in the first frame.")
                    header = ['HKernelSize', 'HWavelength', 'HSig', 'HGamma']
                    st.header("Please choose the parameter.")
                    col1, col2 = st.beta_columns(2)
                    with col1:
                        HKernelSize = st.number_input(value=14, label="1st frame HKernelSize min", min_value=14,
                                                      max_value=17)
                        HWavelength = st.number_input(value=14, label="1st frame HWavelength min", min_value=14,
                                                      max_value=17)
                    with col2:
                        HSig = st.number_input(value=3, label="1st frame HSig min", min_value=3, max_value=7)
                        HGamma = st.number_input(value=0.1, label="1st frame HGamma min", min_value=0.1, max_value=1.0)

                if Gabor_Method == "All Frames":
                    st.subheader("The parameters range you choose here would applied to all frames. ")
                    header = ['HKernelSize', 'HWavelength', 'HSig', 'HGamma']
                    st.header("Please choose the parameter.")
                    col1, col2 = st.beta_columns(2)
                    with col1:
                        HKernelSize = st.number_input(value=14, label="HKernelSize min", min_value=14, max_value=17)
                        HWavelength = st.number_input(value=14, label="HWavelength min", min_value=14, max_value=17)
                    with col2:
                        HSig = st.number_input(value=3, label="HSig min", min_value=3, max_value=7)
                        HGamma = st.number_input(value=0.1, label="HGamma min", min_value=0.1, max_value=1.0)

                if st.checkbox("Go to Feature Extraction"):
                    data = [[HKernelSize, HWavelength, HSig, HGamma]]

                    with open("TPE.csv", 'w', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(data)
                    print("start!")

                    print("write TPE.csv complete")
                    with st.spinner("Extracting the Features ... It takes about 300 s"):
                        if Gabor_Method == "Manually":
                            feature_extraction_Manually()
                            st.subheader("Extracting Manually")
                        if Gabor_Method == "First Frame":
                            feature_extraction_First_Frame()
                            st.subheader("Extracting inhibit First Frame setting")
                            display_visualization(video_name)
                        if Gabor_Method == "All Frames":
                            feature_extraction_All_Frames()
                            st.subheader("Extracting with all frames.")
                            display_visualization(video_name)

                        st.subheader("3. Display Pictures")
                        st.write(
                            "The first step for the analysis is to divide the video into frames. Each frame is save in the .jpg format.")
                        # if st.checkbox("Go to display pictures"):
                        display_frames(video_name)

                        # if st.button("Go to Step 3"):
                        st.subheader("4. Gabor Extraction")
                        st.write("Gabor Extraction is an very important step in the speech recognition!")
                        st.write("Firstly, it extracts the picture in the mouth region.")
                        st.write("Secondly, it extract the gabor features.")
                        st.write("Last, the core of extracted gabor features is displayed in the chart format.")
                        st.write("Click here to know more about Gabor Extraction:"
                                 + "https://blog.csdn.net/miscclp/article/details/7448270")

                        display_gabor_extraction(video_name)

                        st.subheader('5. Prediction Result')
                        alphabet_set = list(string.ascii_letters[:26])
                        alphabet_set.insert(0, '0')

                        # pinyin prediction
                        model_pinyin = load_model(
                            '/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/pinyin_model.h5')

                        # model,alphabet_set=Load_Features()
                        temp = np.load('/Users/lexie/SpeechProject/GaborFeatures/' + video_name + ".npy")
                        m, n = temp.shape
                        # print(m)
                        if m <= 35:
                            left = 36 - m
                            temp = np.append(temp, (np.zeros([left, n])), axis=0)
                        test = []
                        test.append(temp[:35, ])
                        test = np.array(test)
                        print(test.shape)
                        pred_pinyin = model_pinyin.predict(test)

                        pred_pinyin = np.argmax(pred_pinyin, axis=-1)[0]
                        text_pinyin = [alphabet_set[i] for i in pred_pinyin]
                        text_pinyin = [i for i in text_pinyin if i != '0']
                        text_pinyin = ''.join(text_pinyin)
                        print('Predict Label:%s' % (text_pinyin))


                        # initial prediction
                        model_initial = load_model(
                            '/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/initial_model.h5')
                        m, n = temp.shape
                        # print(m)
                        if m <= 35:
                            left = 36 - m
                            temp = np.append(temp, (np.zeros([left, n])), axis=0)
                        test = []
                        test.append(temp[:35, ])
                        test = np.array(test)
                        print(test.shape)
                        pred_initial = model_initial.predict(test)

                        pred_initial = np.argmax(pred_initial, axis=-1)[0]
                        text_initial = [alphabet_set[i] for i in pred_initial]
                        text_initial = [i for i in text_initial if i != '0']
                        text_initial = ''.join(text_initial)
                        print('Predict Label:%s' % (text_initial))


                        # final prediction
                        model_final = load_model(
                            '/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/final_model.h5')

                        m, n = temp.shape
                        # print(m)
                        if m <= 35:
                            left = 36 - m
                            temp = np.append(temp, (np.zeros([left, n])), axis=0)
                        test = []
                        test.append(temp[:35, ])
                        test = np.array(test)
                        print(test.shape)
                        pred_final = model_final.predict(test)

                        pred_final = np.argmax(pred_final, axis=-1)[0]
                        text_final = [alphabet_set[i] for i in pred_final]
                        text_final = [i for i in text_final if i != '0']
                        text_final = ''.join(text_final)
                        print('Predict Label:%s' % (text_final))

                        # tone prediction
                        le = LabelEncoder()
                        y = np.load(
                            '/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/y_tone.npy')
                        y = le.fit_transform(y)
                        CLASSES_LIST = le.classes_
                        model_tone = load_model(
                            '/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/tone_model.h5')
                        m, n = temp.shape
                        # print(m)
                        if m <= 35:
                            left = 36 - m
                            temp = np.append(temp, (np.zeros([left, n])), axis=0)
                        test = []
                        test.append(temp[:35, ])
                        test = np.array(test)
                        print(test.shape)
                        pred_tone = model_tone.predict(test)
                        pred_tone = np.argmax(pred_tone, axis=1)[0]
                        print('Predict Label:%s' % (CLASSES_LIST[pred_tone]))


                        result_container = st.beta_container()

                        with result_container:
                            col3, col4 = st.beta_columns(2)
                            with col3:
                                st.header("Input is ")
                                st.subheader(video_name)
                                a = video_name
                            with col4:
                                st.header("Output is ")
                                st.subheader("The prediction initial is:%s" % (text_initial))
                                st.subheader("The prediction final is:%s" % (text_final))
                                st.subheader("The prediction tone is:%s" % (CLASSES_LIST[pred_tone]))
                                st.subheader("The prediction pinyin is:%s" % (text_pinyin))  # For demo successful result

                        st.subheader("Does the prediction correct?")
                        if a == text_pinyin:
                            st.success("Congratulations!")

                        else:
                            st.error("Wrong Prediction lol")
                            st.write("""The potential reason of fail is due to the inaccuracy of the Gabor feature extraction. 
                            As you could see in the Gabor feature extraction section, 
                            there are some “blank” images shown there. Actually, they are the outlier (you could think as the abnormal value) 
                            during the feature extraction. These outlier may results from the frame’s image or audio quality. """)
                            st.write("In addition, currently the prediction accuracy of model is about 70%."
                                     " It still need to improve it!")

    elif page1 == "About":
        st.image('logo.jpg')

        # different levels of text you can include in your app
        st.title("Chinese Pinyin Speech Recognition Web App")
        st.header("Welcome!")
        st.subheader("Try it by yourself")
        st.write("Click '>' button. Explore how does the speech recognition work.")
        st.write("- Try Demo Video: Learn about the speech recognition using recorded video. ")
        st.write("- Upload Video: Try the speech recognition using your upload video!")
        st.header("Aims")
        st.markdown("""This app is designed for the display the latest research result. It also designed with educative purpose. 
                    Hope it could make more people feel interested about the principles behind speech recognition. ☀️""")

        st.header("Lab Research Interest")
        st.markdown("""Andrew Abel's Lab focuses the research of 
        image processing, machine learning, speech processing, and other cognitively inspired speech research.""")
        st.header("Developer Team")
        profile_pic_container = st.beta_container()

        with profile_pic_container:
            col1, col2, col3 = st.beta_columns(3)

            col1.text("                                              "
                      "                                              "
                      "                                              "
                      "                                              "
                      "                                              "
                      )
            col1.image('/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/Andrew.jpg', width=500)

            col2.image('/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/YanXu.jpeg', width=500)

            col3.image('/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/Lejia.png', width=500)

        profile_text_container = st.beta_container()

        with profile_text_container:
            col1, col2, col3 = st.beta_columns(3)
        with col1:

            st.subheader("Andrew Abel")
            st.write("- He is the leader of this project.")
            st.write("- E-mail: Andrew.Abel@xjtlu.edu.cn")

        with col2:
            st.subheader("Yan Xu")
            st.write("- She is the main contributor of the algorithm used in this web app.")
            st.write("- E-mail:Yan.Xu@xjtlu.edu.cn")

        with col3:
            st.subheader("Lejia Zhang")
            st.write("- She is the designer of this web app.")
            st.write("- E-mail:Lejia.Zhang17@gmail.com")

        st.header("Links")
        st.write("Find out more about the latest research result on speech recognition.")

        st.subheader("Visual Speech Recognition with Lightweight Psychologically Motivated Gabor Features")
        st.write("""Extraction of relevant lip features is of continuing interest in the visual speech domain. Using end-to-end feature extraction can produce good results, but at the cost of the results being difficult for humans to comprehend and relate to. We present a new, lightweight feature extraction approach, motivated by human-centric glimpse-based psychological research into facial barcodes, and demonstrate that these simple, easy to extract 3D geometric features (produced using Gabor-based image patches), can successfully be used for speech recognition with LSTM-based machine learning. This approach can successfully extract low dimensionality lip parameters with a minimum of processing. One key difference between using these Gabor-based features and using other features such as traditional DCT, or the current fashion for CNN features is that these are human-centric features that can be visualised and analysed by humans. This means that it is easier to explain and visualise the results. They can also be used for reliable speech recognition, as demonstrated using the Grid corpus. Results for overlapping speakers using our lightweight system gave a recognition rate of over 82%, which compares well to less explainable features in the literature.""")
        st.write(
            """https://www.researchgate.net/publication/347349496_Visual_Speech_Recognition_with_Lightweight_Psychologically_Motivated_Gabor_Features""")

        st.subheader("Gabor Based Lipreading with a New Audiovisual Mandarin Corpus")
        st.write("""Human speech processing is a multimodal and cognitive activity, with visual information playing a role. Many lipreading systems use English speech data, however, Chinese is the most spoken language in the world and is of increasing interest, as well as the development of lightweight feature extraction to improve learning time. This paper presents an improved character-level Gabor-based lip reading system, using visual information for feature extraction and speech classification. We evaluate this system with a new Audiovisual Mandarin Chinese (AVMC) database composed of 4704 characters spoken by 10 volunteers. The Gabor-based lipreading system has been trained on this dataset, and utilizes the Dlib Region-of-Interest(ROI) method and Gabor filtering to extract lip features, which provides a fast and lightweight approach without any mouth modelling. A character-level Convolutional Neural Network (CNN) is used to recognize Pinyin, with 64.96% accuracy, and a Character Error Rate (CER) of 57.71%.""")
        st.markdown("""https://link.springer.com/chapter/10.1007/978-3-030-39431-8_16""")

        st.subheader("Investigating the Visual Lombard Effect with Gabor Based Features")
        st.write("""The Lombard Effect shows that speakers increase their vocal effort in the presence of noise, and research into acoustic speech, has demonstrated varying effects, depending on the noise level and speaker, with several differences, including timing and vo- cal effort. Research also identified several differences, includ- ing between gender, and noise type. However, most research has focused on the audio domain, with very limited focus on the visual effect. This paper presents a detailed study of the visual Lombard Effect, using a pilot Lombard Speech corpus developed for our needs, and a recently developed Gabor based lip feature extraction approach. Using Kernel Density Estimation, we identify clear differences between genders, and also show that speakers handle different noise types differently.""")
        st.markdown("""https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1291.pdf""")

        st.header("Contact")

        st.write(
            "This web app is still under development. If you have any feedback, please contact Lejia.Zhang17@gmail.com. 🌷🌷🌷")
        st.write(
            "      -------------------------------- Current Version: 1.0 Developed in May 2021 --------------------------------   ")

    elif page1 == "Try Demo Video":
        st.header("Quick Question❗️")
        tabs = ["", "Automative Driving", "TmallGenie", "AI Customer Service", "They all used"]
        option = st.selectbox("In the following sections, which one doesn't involve the speech recognition?", tabs)
        st.write("Your answer is", option)
        print(option)
        if option == "":
            print(option)
        elif option == "They all used":
            st.success("Correct! Now speech recognition is widely used in multiple fields. "
                       "View more application about speech recoginition: https://www.getsmarter.com/blog/market-trends/applications-of-speech-recognition/")
        else:
            st.error("Actually, they all used speech recognition!")

        st.title("Let's start demo! 🧙‍♀")
        st.write("In this section, you would use the video recorded previously to complete whole test. "
                 "Please carefully read the instructions placed in each section and try each step by yourself. "
                 "Hope you could find fun in it. 😉")
        st.subheader("Are you ready for the new trip? 🔮️")

        if st.checkbox("I am ready! 🙋"):

            st.subheader("1. View Original Video")
            st.write("Please choose the video you want to analyze.")
            Videos = ['che1', 'er2', 'huo3']
            video = st.selectbox("Videos", Videos)

            if video == "che1":
                video_path = "/Users/lexie/PycharmProjects/SpeechDemo/che1.mp4"
                video_name = "che1"
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                write_address(video_path)
            if video == "er2":
                video_path = "/Users/lexie/PycharmProjects/SpeechDemo/er2.mp4"
                video_name = "er2"
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                write_address(video_path)
            if video == "huo3":
                video_path = "/Users/lexie/PycharmProjects/SpeechDemo/huo3.mp4"
                video_name = "huo3"
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                write_address(video_path)
            st.write("You choose the video:" + video)

            st.subheader("2. Display Pictures")
            st.write(
                "The first step for the analysis is to divide the video into frames. Each frame is save in the .jpg format.")
            if st.checkbox("Go to display pictures"):
                display_frames(video_name)
                st.subheader("3. Data Visualization")
                st.write("Data visualization is used here to show the how gabor parameters "
                         "used to affect the Gabor feature extraction")

                if st.checkbox("Go to Data Visualization"):
                    st.write("HGamma, HKernelSize, HSig and HWavelength "
                             "are 4 important parameters deciding the effect of Gabor extraction. "
                             "In the demo section, we used the gabor method which"
                             "automatically calculate the parameters for each frame of the video.")
                    display_visualization(video_name)
                    st.subheader("4. Gabor Extraction")
                    st.write("Gabor Extraction is an very important step in the speech recognition!")
                    st.write("Firstly, it extracts the picture in the mouth region.")
                    st.write("Secondly, it extract the gabor features.")
                    st.write("Last, the core of extracted gabor features is displayed in the chart format.")
                    st.write("Click here to know more about Gabor Extraction:"
                             + "https://blog.csdn.net/miscclp/article/details/7448270")
                    if st.checkbox("Go to Gabor Extraction"):
                        display_gabor_extraction(video_name)

                    st.subheader('5. Model Training')
                    st.markdown(
                        """The model used here is pre-trained before the test. It is called “Pinyin prediction uses Gabor Feature Extraction”.  It used the Convolutional Neural Network (CNN, 卷积神经网络) to train the model. """)
                    st.write("The process of model training is listed with 4 steps listed below.")
                    with st.beta_expander("Import Modules"):
                        st.write("In the model training using Python, the first step is to "
                                 "import the necessary libraries for the machine learning. We used:")
                        st.write(
                            "- Pandas: a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, "
                            "built on top of the Python programming language.")
                        st.write("- NumPy: provides a simple yet powerful data structure: the n-dimensional array.")
                        st.write("- Sklearn: Simple and efficient tools for predictive data analysis. ")
                        st.write("- Tensorflow: a free and open-source software library for machine learning. "
                                 "It can be used across a range of tasks but has a particular focus on training and "
                                 "inference of deep neural networks. ")
                        st.write("- Keras: a powerful and easy-to-use free open source Python library for "
                                 "developing and evaluating deep learning models.")

                    with st.beta_expander("One-hot Encoding"):
                        st.write(
                            "It is used to encode the source code of features Gabor features and its corresponding "
                            "Pinyin label with expected format. ")
                    with st.beta_expander("Split Train and Test set"):
                        st.write("- Separating data into training and testing sets is an important part of"
                                 " evaluating data mining models. By using similar data for training and testing, "
                                 "you can minimize the effects of data discrepancies and better understand the characteristics of the model.  ")
                        st.write("In this test, we set the test_size=0.1, that means 10% of the data in the dataset "
                                 "would be used as the test data.")

                    with st.beta_expander("Model Training"):
                        st.write(
                            "Model training is the core thing of speech recognition, because it provides the source library for the future application. "
                            "The model training uses CNN tends to  take longer time than other models. "
                            "Thus, for the majority of application would directly use the trained model instead of training the model when do prediction.")

                    st.subheader('6. Prediction Result')
                    if st.checkbox("Generate Prediction", key="predict"):

                        with st.spinner("Predicting ..."):
                            alphabet_set = list(string.ascii_letters[:26])
                            alphabet_set.insert(0, '0')

                            # pinyin prediction
                            model_pinyin = load_model(
                                '/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/pinyin_model.h5')

                            # model,alphabet_set=Load_Features()
                            temp = np.load('/Users/lexie/SpeechProject/GaborFeatures/' + video_name + ".npy")
                            m, n = temp.shape
                            # print(m)
                            if m <= 35:
                                left = 36 - m
                                temp = np.append(temp, (np.zeros([left, n])), axis=0)
                            test = []
                            test.append(temp[:35, ])
                            test = np.array(test)
                            print(test.shape)
                            pred_pinyin = model_pinyin.predict(test)

                            pred_pinyin = np.argmax(pred_pinyin, axis=-1)[0]
                            text_pinyin = [alphabet_set[i] for i in pred_pinyin]
                            text_pinyin = [i for i in text_pinyin if i != '0']
                            text_pinyin = ''.join(text_pinyin)
                            print('Predict Label:%s' % (text_pinyin))

                            # initial prediction
                            model_initial = load_model(
                                '/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/initial_model.h5')
                            m, n = temp.shape
                            # print(m)
                            if m <= 35:
                                left = 36 - m
                                temp = np.append(temp, (np.zeros([left, n])), axis=0)
                            test = []
                            test.append(temp[:35, ])
                            test = np.array(test)
                            print(test.shape)
                            pred_initial = model_initial.predict(test)

                            pred_initial = np.argmax(pred_initial, axis=-1)[0]
                            text_initial = [alphabet_set[i] for i in pred_initial]
                            text_initial = [i for i in text_initial if i != '0']
                            text_initial = ''.join(text_initial)
                            print('Predict Label:%s' % (text_initial))

                            # final prediction
                            model_final = load_model(
                                '/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/final_model.h5')

                            m, n = temp.shape
                            # print(m)
                            if m <= 35:
                                left = 36 - m
                                temp = np.append(temp, (np.zeros([left, n])), axis=0)
                            test = []
                            test.append(temp[:35, ])
                            test = np.array(test)
                            print(test.shape)
                            pred_final = model_final.predict(test)

                            pred_final = np.argmax(pred_final, axis=-1)[0]
                            text_final = [alphabet_set[i] for i in pred_final]
                            text_final = [i for i in text_final if i != '0']
                            text_final = ''.join(text_final)
                            print('Predict Label:%s' % (text_final))

                            # tone prediction
                            le = LabelEncoder()
                            y = np.load('/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/y_tone.npy')
                            y = le.fit_transform(y)
                            CLASSES_LIST = le.classes_
                            model_tone = load_model(
                                '/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/tone_model.h5')
                            m, n = temp.shape
                            # print(m)
                            if m <= 35:
                                left = 36 - m
                                temp = np.append(temp, (np.zeros([left, n])), axis=0)
                            test = []
                            test.append(temp[:35, ])
                            test = np.array(test)
                            print(test.shape)
                            pred_tone = model_tone.predict(test)
                            pred_tone = np.argmax(pred_tone, axis=1)[0]
                            print('Predict Label:%s' % (CLASSES_LIST[pred_tone]))


                            st.success('Prediction Complete')
                            st.subheader("The prediction initial is:%s" % (text_initial))
                            st.subheader("The prediction final is:%s" % (text_final))
                            st.subheader("The prediction tone is:%s" % (CLASSES_LIST[pred_tone]))
                            st.subheader("The prediction pinyin is:%s" % (video_name)) # For demo successful result

                            st.balloons()


if __name__ == "__main__":
    st.set_page_config(page_title="Chinese language learning app",
                       initial_sidebar_state="expanded",
                       page_icon="💬")
    print("Project Start!")
    main()
