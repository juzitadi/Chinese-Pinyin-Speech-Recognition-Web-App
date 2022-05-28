# 0-- Import Libraries
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from subprocess import call
import pandas as pd
import altair as alt
video_name=""
def get_data(filename):
    test_data = pd.read_csv(filename)

    return test_data
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

# 2-- Build the structure of app

# Streamlit apps can be split into sections

# container -> horizontal sections
# columns -> vertical sections (can be created inside containers or directly in the app)
# sidebar -> a vertical bar on the side of your app


header_container = st.beta_container()
stats_container = st.beta_container()

# You can place things (titles, images, text, plots, dataframes, columns etc. ) inside container
with header_container:
    # for example a logo or a image that looks like a website header
    st.image('logo.jpg')

    # different levels of text you can include in your app
    st.title("Chinese language learning app")
    st.header("Welcome!")
    st.subheader("Amazing work")
    st.write("Try it by yourself")



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

col1, col2, col3 = st.beta_columns(3)
with col1:
    if st.button('che1'):
        st.write('1')
        video_file = open('che1.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        # video_path = "/Users/lexie/SpeechProject/WordVideo/*.mp4"
        # call(["python", "/Users/lexie/PycharmProjects/SpeechDemo/FeatureExtraction/Main.py"])
        #
        #
        # print("complete")
        test_data = get_data("/Users/lexie/SpeechProject/Qs6/ASheet/huo3/01.csv")
        st.write(test_data.head())
        st.line_chart(test_data)

with col2:
    if st.button("er2"):
        st.write('2')
        video_file = open('er2.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        video_name = "che1"
        call(["python", "test.py"])
with col3:
    if st.button("huo3"):
        st.write('3')
        video_file = open('huo3.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        video_name = "che1"
        call(["python", "test.py"])
# else:
#     st.write('Goodbye')
# 5-- Feature extraction of the recorded video

# 6-- Model training and testing on the user input

# 7-- Data visualization

