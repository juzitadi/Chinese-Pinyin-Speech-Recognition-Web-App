# 0-- Import Libraries
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from subprocess import call
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
import os
import csv
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import keras
from keras.utils import plot_model,to_categorical
from keras import Model,Input,regularizers
from keras.layers import Reshape,Flatten
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
import keras.callbacks as kcallbacks
from sklearn.preprocessing import LabelEncoder
import string
from keras.optimizers import SGD
import warnings
import tensorflow as tf
# from lsuv_init import LSUVinit
warnings.filterwarnings('ignore')
from tensorflow.keras import layers
from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply,BatchNormalization,MaxPool1D
from keras.models import Model
import time

# from attention_utils import get_activations
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

import  pandas as pd
import  numpy as np
from keras.wrappers import scikit_learn

# video_path=""
def get_data(filename):
    test_data = pd.read_csv(filename)

    return test_data

def Load_Features():
    X_features=np.load("/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/X_gabor.npy")
    y=np.load("/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/y_pinyin.npy")

    alphabet_set=list(string.ascii_letters[:26])
    alphabet_set.insert(0,'0')
    print("alphabet_set length",len(alphabet_set))

    # One-hot Encoding
    y=to_categorical(y,num_classes=27)

    # split train and test
    np.random.seed(116)
    np.random.shuffle(X_features)
    np.random.seed(116)
    np.random.shuffle(y)
    # tf.random.set_seed(116)
    tf.set_random_seed(116)
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.1, random_state=2019)
    print("X_train shape",X_train.shape)


    # Models
    NUM_CLASSES = 27
    BATCH_SIZE = 128
    EPOCHS = 150
    MODEL_SAVE_PATH = '/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/model/pinyin_gabor_cnn1.h5'
    inputs = Input(shape=(35, 7))
    SINGLE_ATTENTION_VECTOR = False

    def attention_3d_block(inputs, single_attention_vector=False):
        # 如果上一层是LSTM，需要return_sequences=True
        # inputs.shape = (batch_size, time_steps, input_dim)
        time_steps = K.int_shape(inputs)[1]
        input_dim = K.int_shape(inputs)[2]
        a = Permute((2, 1))(inputs)
        a = Dense(time_steps, activation='softmax', name='attention')(a)
        if single_attention_vector:
            a = Lambda(lambda x: K.mean(x, axis=1))(a)
            a = RepeatVector(input_dim)(a)

        a_probs = Permute((2, 1))(a)
        # 乘上了attention权重，但是并没有求和，好像影响不大
        # 如果分类任务，进行Flatten展开就可以了
        # element-wise
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul

    def attention_model():
        inputs = Input(shape=(35, 7))
        x = BatchNormalization()(inputs)
        x = Conv1D(filters=32, kernel_size=1, padding="same", kernel_initializer='he_normal',
                   bias_initializer='zeros', activation='relu')(x)  # , padding = 'same'
        x = MaxPool1D()(x)
        # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
        # 对于GPU可以使用CuDNNLSTM
        lstm_out = Bidirectional(LSTM(32, return_sequences=True))(x)
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(lstm_out)
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(lstm_out)
        attention_mul = attention_3d_block(lstm_out)
        attention_mul = MaxPool1D(pool_size=8, strides=2)(attention_mul)
        output = Dense(27, activation='sigmoid')(attention_mul)
        model = Model(inputs=[inputs], outputs=output)
        return model

    model = attention_model()
    print("model.summary",model.summary())

    # Take long time **********************
    a = time.time()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lr_reduce = kcallbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1, min_lr=0.00001)
    save_model = kcallbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1,
                                            save_best_only=True)
    callback_list = [save_model, lr_reduce]
    history = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test),
                        callbacks=callback_list)
    b = time.time()
    print("time=", b - a)

    temp = np.load('/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/WordRecognition/che1.npy')
    m, n = temp.shape
    # print(m)
    if m <= 35:
        left = 36 - m
        temp = np.append(temp, (np.zeros([left, n])), axis=0)
    test = []
    test.append(temp[:35, ])
    test = np.array(test)
    print(test.shape)
    pred = model.predict(test)

    pred = np.argmax(pred, axis=-1)[0]
    text = [alphabet_set[i] for i in pred]
    text = [i for i in text if i != '0']
    text = ''.join(text)
    print('Predict Label:%s' % (text))

# 3-- Collect video input from the user

# webrtc_streamer(key="example")


# 4-- Click the "Process" button
# def call_analysis():

# if st.button('Process'):
#     st.write('Ready to process')
#     call(["python", "/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/Main.py"])
#     print("complete")
#     test_data=get_data("/Users/lexie/SpeechProject/Qs6/ASheet/huo3/01.csv")
#     st.write(test_data.head())
# def ChooseButton():
#
#         return path


def write_address(file_path):
    with open("test.txt", "w")as f:
        f.write(file_path)

# else:
#     st.write('Goodbye')
# 5-- Feature extraction of the recorded video

# 6-- Model training and testing on the user input

# 7-- Data visualization
def feature_extraction():
    call(["python", "/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/Main.py"])
    print("complete")

def main():
    st.set_page_config(page_title="Chinese language learning app",
                       initial_sidebar_state="expanded",
                       page_icon="💬")
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



    tabs = ["Try Demo Video", "Record Video","Upload Video","About"]
    page1=st.sidebar.selectbox("Page",tabs)
    # page2=st.sidebar.button("About")


    if page1=="Record Video":
        st.header("Record Video")
    elif page1=="Upload Video":
        st.file_uploader('')
    elif page1=="About":
        st.image('logo.jpg')

        # different levels of text you can include in your app
        st.title("Chinese language learning app")
        st.header("Welcome!")
        st.subheader("Amazing work")
        st.write("Try it by yourself")
    elif page1=="Try Demo Video":
        # st.header("1. View Original Video")
        # st.text("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # Videos = ['che1', 'er2', 'huo3']
        # video = st.selectbox("Videos", Videos)
        #
        # if video == "che1":
        #     video_path = "/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/che1.mp4"
        #     video_file = open(video_path, 'rb')
        #     video_bytes = video_file.read()
        #     st.video(video_bytes)
        #     write_address(video_path)
        # if video == "er2":
        #     video_path = "/Users/lexie/PycharmProjects/SpeechDemo/er2.mp4"
        #     video_file = open(video_path, 'rb')
        #     video_bytes = video_file.read()
        #     st.video(video_bytes)
        #     write_address(video_path)
        # if video == "huo3":
        #     video_path = "/Users/lexie/PycharmProjects/SpeechDemo/huo3.mp4"
        #     video_file = open(video_path, 'rb')
        #     video_bytes = video_file.read()
        #     st.video(video_bytes)
        #     write_address(video_path)
        #
        # # if st.button("Go to Step 2"):
        # st.header("2. Display Pictures")
        # st.text("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # images = [cv2.imread(file) for file in glob.glob("/Users/lexie/SpeechProject/Qs6/APicture/che1/*.jpg")]
        # # st.image(images, use_column_width=True, caption=["some generic text"] * len(images))
        # idx = 0
        # for _ in range(len(images) - 1):
        #     cols = st.beta_columns(4)
        #
        #     if idx < len(images):
        #         cols[0].image(images[idx], width=550, caption=idx)
        #     idx += 1
        #
        #     if idx < len(images):
        #         cols[1].image(images[idx], width=550, caption=idx)
        #     idx += 1
        #
        #     if idx < len(images):
        #         cols[2].image(images[idx], width=550, caption=idx)
        #     idx += 1
        #     if idx < len(images):
        #         cols[3].image(images[idx], width=550, caption=idx)
        #         idx = idx + 1
        #     else:
        #         break
        #
        #     # if st.button("Go to Step 3"):
        # st.header("3. Gabor Extraction")
        # st.text("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # images = [cv2.imread(file) for file in glob.glob("/Users/lexie/SpeechProject/Qs6/APicture/che1/*.jpg")]
        # mouths = [cv2.imread(file) for file in glob.glob("/Users/lexie/SpeechProject/Qs6/Amouth/che1/*.jpg")]
        # gabors = [cv2.imread(file) for file in glob.glob("/Users/lexie/SpeechProject/Qs6/AGabor/che1/*.jpg")]
        # features = [cv2.imread(file) for file in
        # glob.glob("/Users/lexie/SpeechProject/Qs6/AFeatures/che1/*.png")]
        # idx = 0
        # for _ in range(len(images) - 1):
        #     cols = st.beta_columns(4)
        #
        #     if idx < len(images):
        #         cols[0].image(images[idx], width=350, caption=idx)
        #
        #     if idx < len(images):
        #         cols[1].image(mouths[idx], width=350, caption=idx)
        #
        #     if idx < len(images):
        #         cols[2].image(gabors[idx], width=350, caption=idx)
        #
        #     if idx < len(images):
        #         cols[3].image(features[idx], width=350, caption=idx)
        #         idx = idx + 1
        #
        #     else:
        #         break
            # if st.button("Go to Step 4"):
        st.header("4. Data Visualization")
        st.text("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        info=os.listdir('/Users/lexie/SpeechProject/Qs6/ATPE/che1')
        info=sorted(info)
        df = pd.DataFrame([], columns=['HGamma', 'HKernelSize', 'HSig', 'HWavelength','ID'])
        for i in info:
            data = pd.read_csv('/Users/lexie/SpeechProject/Qs6/ATPE/che1/%s' % i, header=None)
            data.columns = ['name', 'value']
            data = data.set_index(['name'])
            data = data.T
            df = df.append(data)
        df = df.reset_index()
        df['ID'] = np.arange(len(df))
        line_chart_data=df.copy()
        # cross_tab=pd.crosstab(x=len(line_chart_data.index),y=line_chart_data["Box_width"])
        fig1 = px.line(x=line_chart_data["ID"],y=line_chart_data["HGamma"])
        fig2 = px.line(x=line_chart_data["ID"], y=line_chart_data["HKernelSize"])
        fig3 = px.line(x=line_chart_data["ID"], y=line_chart_data["HSig"])
        fig4 = px.line(x=line_chart_data["ID"], y=line_chart_data["HWavelength"])



        fig1.update_layout(
            width=400,
            height=300,
        )

        fig2.update_layout(
            width=400,
            height=300
        )
        fig3.update_layout(
            width=400,
            height=300
        )
        fig4.update_layout(
            width=400,
            height=300
        )

        interactive_container = st.beta_container()

        with interactive_container:
            col1,col2 = st.beta_columns(2)
            with col1:
                st.write(fig1)
                # HGamma=st.slider("Select the number of HGamma", 7, 17)
                HGamma = st.number_input(value=7, label="HGamma min", min_value=7, max_value=17)
                st.write(fig2)
                # HKernelSize=st.slider("Select the number of HKernelSize", 7, 17)
                HKernelSize=st.number_input(value=7, label="HKernalSize min", min_value=7, max_value=17)

            with col2:
                st.write(fig3)
                # HSig=st.slider("Select the number of HSig", 3, 7)
                HSig=st.number_input(value=3, label="HSig min", min_value=3, max_value=7)
                st.write(fig4)
                # HWavelength=st.slider("Select the number of HWavelength", 0.1, 1.0)
                HWavelength =st.number_input(value=0.1, label="HWavelength min", min_value=0.1, max_value=1.0)

            # 可以write in column
            header = ['HKernelSize','HWavelength','HSig','HGamma' ]
            data = [{'HKernelSize': HKernelSize, 'HWavelength': HWavelength,'HSig': HSig, 'HGamma': HGamma}]

            with open("TPE.csv", 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, header)
                writer.writeheader()
                writer.writerows(data)
            print("start!")
            call(["python", "/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/Main.py"])
            print("write TPE.csv complete")


            # if(HGamma!=7 | HKernelSize!=7 | HSig !=3 | HWavelength!=0.1):
            # if (HGamma != 7):
            #     print("start!")
            #     call(["python", "/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/Main.py"])

            # with open('TPE.csv', encoding='utf-8') as f:
            #     reader = csv.reader(f)
            #     header = next(reader)
            #     print(header)
            #     for row in reader:
            #         print(row)



        # plt.plot(df.Box_width, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='Box_width')
        # plt.legend(loc="upper right")
        # plt.xlabel('csv')
        # plt.ylabel('value')
        # plt.show()


        st.header('5. Model Training')
        st.write("In this section it is possible to do cross-validation of the model.")
        with st.beta_expander("Explanation"):
            st.markdown(
                    """The Prophet library makes it possible to divide our historical data into training data and testing data for cross validation. The main concepts for cross validation with Prophet are:""")
            st.write(
                    "Training data (initial): The amount of data set aside for training. The parameter is in the API called initial.")
            st.write("Horizon: The data set aside for validation.")
            st.write("Cutoff (period): a forecast is made for every observed point between cutoff and cutoff + horizon.""")

        with st.beta_expander("Split Train and Test set"):
            initial = st.number_input(value=365, label="initial", min_value=30, max_value=1096)
            initial = str(initial) + " days"

            period = st.number_input(value=90, label="period", min_value=1, max_value=365)
            period = str(period) + " days"

            horizon = st.number_input(value=90, label="horizon", min_value=30, max_value=366)
            horizon = str(horizon) + " days"

            st.write(
                    f"Here we do cross-validation to assess prediction performance on a horizon of **{horizon}** days, starting with **{initial}** days of training data in the first cutoff and then making predictions every **{period}**.")
            st.markdown(
                    """For more information read the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)""")
        with st.beta_expander("Models"):
            st.markdown(
                    """The Prophet library makes it possible to divide our historical data into training data and testing data for cross validation. The main concepts for cross validation with Prophet are:""")
            st.write(
                    "Training data (initial): The amount of data set aside for training. The parameter is in the API called initial.")
            st.write("Horizon: The data set aside for validation.")

            initial1 = st.number_input(value=365, label="initial1", min_value=30, max_value=1096)
            initial1 = str(initial) + " days"

            period1 = st.number_input(value=90, label="period1", min_value=1, max_value=365)
            period1= str(period) + " days"

            horizon1 = st.number_input(value=90, label="horizon1", min_value=30, max_value=366)
            horizon1 = str(horizon) + " days"

        with st.beta_expander("Plot Curve"):
            st.write(fig1)
            # Load_Features()
            # print("Complete")

        st.header('6. Prediction Result')
        result_container=st.beta_container()

        with result_container:
            col3,col4=st.beta_columns(2)
            with col3:
                st.header("Input is ")
                st.subheader("che1")
                a="che1"
            with col4:
                st.header("Prediction is")
                st.subheader("che1")
                b="che1"
            if a==b:
                st.success("Correct Prediction!")
                if st.button("Success"):
                    st.balloons()
                if st.button("Fail"):
                    st.subheader("Potential Wrong Reason")
                    st.markdown(
                        """The Prophet library makes it possible to divide our historical data into training data and testing data for cross valida""")

            else:
                st.error("Wrong Prediction lol")
                if st.button("Fail"):
                    st.subheader("Potential Wrong Reason")
                    st.markdown(
                        """The Prophet library makes it possible to divide our historical data into training data and testing data for cross valida""")







# header_container = st.beta_container()
# stats_container = st.beta_container()
#
# # You can place things (titles, images, text, plots, dataframes, columns etc. ) inside container
#
# with header_container:
#     # for example a logo or a image that looks like a website header
#






# 2-- Build the structure of app

# Streamlit apps can be split into sections

# container -> horizontal sections
# columns -> vertical sections (can be created inside containers or directly in the app)
# sidebar -> a vertical bar on the side of your app



# path = button_choose()
# # print("path=", path)
# if (path != " "):
#     print(0)
#     print("VideoPath=", path)
#     feature_extraction()
# print("1",button_choose())
if __name__=="__main__":
    main()
# path=button_choose()
# print(path)