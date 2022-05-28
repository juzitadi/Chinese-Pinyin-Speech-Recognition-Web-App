# Chinese-Pinyin-Speech-Recognition-Web-App
Writer: Lejia Zhang 2021/06/17


0. The Python version of this project is Python 3.7. Related modules should also installed using pip install or python3 -m install before running the program. 

1. Streamlit 
Streamlit is a novel framework which could easily used in Python as a library. This project was written with Streamlit. Since currently there are few people using Streamlit, I advised you could search more useful resources on Google or YouTube.  

These resource materials are useful for development: 
Tutorial: https://docs.streamlit.io/en/stable/index.html

API: https://docs.streamlit.io/en/stable/api.html

2. The main interface is written in the WebApp.py in the FeatureExtraction folder. 


3. File used in the project 

(1) Video file: There are 6 video files used in this project: che1.mp4, er2.mp4, huo3.mp4, ru4.mp4, san1.mp4 and shang4.mp4. They are stored in the SpeechDemo folder. If you want to test more video files, you could change corresponding code in WebApp.py. 

(2) FeatureExtraction and related file
In this project, there are 3 methods used for feature extraction, which are in 3 folders respectively (FeatureExtraction, FE_manually and FeatureExtraction-1st_Frame).

Since in this project we need to display the feature extraction results, you should change the paths in FeatureExtraction/Main.py, FE_manually/Main.py and FeatureExtraction-1st_Frame/Main.py to store the feature extraction results. 

Also, the paths in WebApp.py should also be changed to display the results. 

(3) Model file
The model used in this project is pertained using gabor_cnn_final.ipynb, gabor_cnn_initial.ipynb, gabor_pinyin_1.ipynb and gabor_cnn_tone.ipynb. 

The trained models are intial_model.h5, final_model.h5, tone_model.h5 and pinyin_model.h5 stored in SpeechDemo/FeatureExtraction/WordRecognition folder. In the WebApp.py, you could change paths in your computer and load them directly. 


4. Contact

You could send e-mail to me for getting more details about this project. Thanks!

Lejia.Zhang17@gmail.com
