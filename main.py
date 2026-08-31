import os
import pickle
import time

from io import BytesIO
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import tensorflow_hub as hub
from bokeh.models import CustomJS
from bokeh.models.widgets import Button
from pydub import AudioSegment
from python_speech_features import mfcc
from scipy.io import wavfile
from st_btn_select import st_btn_select
from streamlit_bokeh_events import streamlit_bokeh_events
from streamlit_option_menu import option_menu
from tqdm import tqdm
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from utils import ResNet50,entry_flow,middle_flow,exit_flow,lstm,show_plot,metric_df,plot_signals,CustomCallback
from experiment.cfg import Config

df = pd.read_csv("experiment/audio_data.csv")
classes = list(df["emotion"].unique())
model = None
config = None
p_path = os.path.join("experiment/pickles","conv.p")
with open("experiment/pickles/conv2.p","rb") as handle:
    config = pickle.load(handle)

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1,center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
def classify(uploaded_file,sltd_model):
    if uploaded_file != 1:
        st.write(uploaded_file)
                            
        file_var = AudioSegment.from_wav(uploaded_file)
        file = "filename.wav"
        file_var.export(file, format='wav')

    wav,rate = librosa.load("filename.wav",sr=16000)

    mask = envelope(wav,rate,0.0005)
    wav = wav[mask[0]]
                            
    # p_path = os.path.join("experiment/pickles","conv.p")
    model_path = "experiment/models/"
    # with open(p_path,"rb") as handle:
    #     config = pickle.load(handle)
    models = {"Xception":"Xception100M.model","ResNet-50":"res50100M.model","LSTM":"LSTM100M.model"}

    model = tf.keras.Sequential([
        hub.KerasLayer(f'{model_path}{models[sltd_model]}')])
    if sltd_model == "LSTM":
        model.build((1,9,13))
    else:
        model.build((1,9,13,1))   
                            #model = tf.keras.models.load_model('experiment/models/conv.model')

                            # Check its architecture
    model.summary()

    pred_s = []
    signals = []
    y_prob = []
    for i in tqdm(range(0,wav.shape[0]-config.step,config.step)):
        sample = wav[i:i+config.step]
        x = mfcc(sample,rate,numcep=config.nfeat,
                nfilt=config.nfilt,nfft=config.nfft)
        x = (x-config.min)/(config.max - config.min)
                                        
        if config.mode == "conv":
            x = x.reshape(1,x.shape[0],x.shape[1],1)
        elif config.mode == "time":
            x = np.expand_dims(x_axis=0)  # type: ignore

        signals.append(sample)
        y_hat = model.predict(x)
        pred_s.append(y_hat)
        y_prob.append(y_hat)

    fn_prob = np.mean(y_prob,axis=0).flatten()

    class_dict = {'neutral':[], 'calm':[], 'happy':[], 'sad':[], 'angry':[], 'fearful':[], 'disgust':[], 'surprised':[]}
    for c,p in zip(classes,fn_prob):
        class_dict[c].append(p)
    return class_dict,fn_prob,signals,pred_s

def showResults(class_dict,fn_prob,signals,pred_s):
    
    st.dataframe(pd.DataFrame(class_dict))
    st.write("Classification:", classes[np.argmax(fn_prob)])
    
    pred_s = [classes[np.argmax(pred)] for pred in pred_s]
    st.pyplot(plot_signals(signals,pred_s))


def main():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)
    
    data_dir = "../../RAVDESS"
    save_dir = "../dataset/"
    clean_dir = save_dir+"clean/"
    st.set_page_config(layout="wide")
    navbar = st.sidebar
    title = """# SPEECH EMOTION RECOGNITION SYSTEM"""

    #Initialize CSS
    local_css("./style.css")
        
    with navbar:
        selected = option_menu(
            
            menu_title = "SERS",
            
            options = ["HOME / INTRO", 
                        "DATA",
                        "MODELS",
                        "EXPERIMENTS",
                        "MODEL TESTING",
                        "SYSTEM DEMO"
                    ],
            
            icons  =  ["house-door-fill",
                    "hdd-stack-fill",
                    "gear-fill",
                        "bar-chart-line-fill",
                        "wrench",
                        "lightning-fill"
                    ],
            
            styles =
            {
                "container": { "background-color": "#001330", "border-radius": 0},
                "menu-title": {"font-weight": "bold", "font-family": "sans-serif", "color": "#FFD233", "text-align": "center", "font-size": "40px"},
                "menu-icon": {"display": "none"},
                "icon": {"color": "#E4DB00", "font-size": "25px"},
                "nav-link": {"color": "white","font-family": "sans-serif", "font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#01d4d"},
                "nav-link-selected": {"font-weight": "normal", "background-color": "#001d4d", 'color': '#FFD233'}
            }
        )
        
        def redirect(link):
            title = link
            return title
        
    redr = redirect(selected)
    st.markdown(f""" 
                    # {str(redr)}
                    ---
                """)

    match redr:
        case "HOME / INTRO":
            col1, col2, col3 = st.columns([.6, 0.2, 2.5])
            
            with col1:
                choice = st.selectbox("SELECT OVERVIEW", ("Abstract", "Problem Statements", "Hypotheses"))
                
            with col3:
                st.header(choice)
                match choice.upper():
                    case "ABSTRACT":
                        st.write("""
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In recent years, technology has been evolving at an exponential rate and in turn, is providing more 
                        and more benefits and convenience to the public. This ranges from Social medias, Virtual Reality, internet of things 
                        and, most notably, Artificial Intelligence. The researchers believe that the next step in improving technology is through
                        emotionally aware systems. In doing so, the researchers contribute to Speech Emotion Recognition (SER) research by studying
                        the performance of three deep learning models, Xception, Residual Networks - 50 and Long Short-Term memory, in classifying
                        speech emotions using the RAVDESS dataset. The study uses Mel-Frequency Cepstral Coefficients(MFCC) for the features of each data
                        to train the different deep learning models. The researchers found that Xception performed the best at 77% \accuracy, while 
                        ResNet-50 came close at 75% \accuracy and LSTM performing at only 62% \accuracy. The study concludes that there is a significant 
                        difference between the accuracy of the models and that their is significant evidence that shows relationship between the deep learning
                        models and the emotion types.

                        <b>Keywords: Artificial Intelligence, Speech Emotion Recognition, Emotion, RAVDESS, Deep Learning, Audio Classification </b>
                        """,unsafe_allow_html = True)
                    case "PROBLEM STATEMENTS":
                        st.markdown("""
                                    ##### The study aims to measure and identify characteristics of the following:
                                    <ol>
                                        <li><b>The accuracy of deep learning models in recognizing speech emotion:</b>
                                            <ul>
                                                <li>Deep Boltzmann Machine (DBM)
                                                <li>Long Short-Term Memory (LSTM)
                                                <li>Residual Neural Network - 50 (ResNet-50)
                                            </ul>
                                        <li><b>The difference in the accuracy of the deep learning models.</b>
                                        <li><b>The difference in the F1-score and the emotion type which includes:</b>
                                            <ul>
                                            <li>Anger
                                            <li>Fear
                                            <li>disgust
                                            <li>Joy
                                            <li>Sadness
                                            <li>Surprise
                                            <li>Neutral
                                            </ul>
                                    </ol>
                                    """, unsafe_allow_html=True)
                    case "HYPOTHESES":
                        st.markdown("""
                                    ##### Prior to capturing the findings in the study, the following are the stated claims:
                                    <ul>
                                        <li>There is no significant difference in the accuracy and deep learning models.
                                        <li>There is no significant difference between the precision and recall of the deep learning models.
                                    </ul>
                                    """, unsafe_allow_html=True)
                        
        case "DATA":
            with st.container():
                col1, col2, col3, col4 = st.columns([2, .25 ,.75, .25])
                with col1:
                    st.header("RAVDESS")
                    st.markdown("""
                                Based on the findings of <b>Livingstone & Russo (2018)</b>,it is a gender balanced dataset consisting 
                                of speech and song statements also expressed in two intensities and an additional neutral format. 
                                The total set of  7356 recordings were each rated 10 times based on the following criteria: 
                                emotional validity, intensity, and genuineness. Two hundred and forty-seven North American individuals 
                                who were untrained research participants provided the ratings gathered. A further set of 72 participants 
                                were responsible for the provision of test-retest data.
                                """, unsafe_allow_html=True)
                with col3:
                    st.header('')
                    emo_sltd = st.selectbox("Select Emotion",
                                            ('Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise'))
                    sample_src = "./sample wav audio/"
                    emo_audio = ""
                    
                    #LINKS TO BE REPLACED WITH WAV FILES WITHIN REPO
                    match emo_sltd:
                        case 'Neutral':   
                            emo_audio = open(sample_src + 'neutral.wav', 'rb')
                        case 'Happy':
                            emo_audio = open(sample_src + 'happy.wav', 'rb')
                        case 'Sad':
                            emo_audio = open(sample_src + 'sad.wav', 'rb')
                        case 'Angry':
                            emo_audio = open(sample_src + 'angry.wav', 'rb')
                        case 'Fear':
                            emo_audio = open(sample_src + 'fear.wav', 'rb')
                        case 'Disgust':
                            emo_audio = open(sample_src + 'disgust.wav', 'rb')
                        case 'Surprise':
                            emo_audio = open(sample_src + 'surprise.wav', 'rb')                 
                            
                    st.audio(emo_audio)
                
            with st.container():  
                col1, col2, col3, col4 = st.columns([2, .1 ,1, .1])
                with col1:
                    #SAMPLE data graph; TO BE UPDATED
                    # chart_data = pd.DataFrame(
                    # np.random.randn(20, 3),
                    # columns=['a', 'b', 'c'])
                    # st.line_chart(chart_data)
                    st.image("experiment/audio_wave.png","Audio by Emotion Visualized")
                    st.image("experiment/mfccs.png","Features extracted (MFCC) by emotion")
                with col3:
                    st.header('Data and Preprocessing')
                    st.write("""
                    The RAVDESS dataset is categorized into eight different emotions, 
                    neutral, calm, happy, sad, angry, disgust, fearful and surprised. Looking at the visualization, you can see
                    just how different each emotions are from each other. This, however, by itself is not enough for a deep learning
                    model to properly learn it. In improving the features, the researchers extracted Mel-Frequency Cepstral Coefficients
                    from each audio data. It's visualization can be seen at the figure below. These features are then what is ultimately 
                    used to train the deep learning models in the study.
                    """)
                            
        case "MODELS":
            c1 = st.container()
            c2 = st.container()
            col1, col2 = st.columns([.5,2])
            sltd_model = ""
            
            with c1:            
                with col1:
                    sltd_model = st.selectbox("Select Model", ("Xception", "LSTM", "ResNet-50"))
                    model_desc = ""
                    model_feats = ""
                    model_finds = ""
                    
                with col2:
                    match sltd_model:
                        case "Xception":
                            st.header("Xception")                    
                            model_desc = ("""
                                            Xception is a convolutional neural network developed by improving on the inception modules 
                                            by 1x1 convolutions to each channel and a 3x3 convolution to each output. Doing this introduced 
                                            depthwise separable convolutions to deep learning and was popularized when the paper, 
                                            Xception: Deep Learning with Depthwise Separable, was published. This change pushed inception modules 
                                            to the extreme, hence, Xception. It is developed by Francois Chollet, who also developed keras.
                                        """)
                            model_feats = (f"""
                                           <ul>
                                           <li><h5>Introduced depthwise separable convolutions to Inception mdules.
                                                <ul>
                                                    <li><p>Chollet (2019).
                                                </ul>
                                            </li>
                                            <li><h5>Is a pretrained network that can classify data 1000 into object categories.
                                                <ul>
                                                    <li><p>Nutan. (2021).
                                                </ul>
                                            </li>
                                            <li><h5>High accuracy and fast training speed in facial emotion recognition.
                                                <ul>
                                                     <li><p>Kim, Poulose & Han. (2021).
                                                </ul>
                                            </li>
                                            </ul>
                                           """)
                            
                            model_finds = ("""
                                          <ol>
                                            <li><h5>Reached 100% training accuracy in birds sounds recognition.  
                                                <ul>
                                                     <li><p>Kumar, Gupta & Singh. (2014)
                                                </ul>
                                            </li>
                                            <li><h5>Introduced improvements in Audio Event Detection and Tagging
                                                <ul>
                                                     <li><p> Gajarsky & Purwins (2018).
                                                </ul>
                                            </li>
                                            <li><h5>Performed better than baseline in classifying environmental audio.
                                                <ul>
                                                     <li><p> Chhikara (2021).
                                                </ul>
                                            </li>
                                          </ul>
                                        """)
                        case "LSTM":
                            st.header("Long Short-Term Memory (LSTM)")
                            model_desc = ("""
                                            Developed by Hochreiter and Schmidhuber in 1997. 
                                            It is a recurrent neural network that is capable of learning long-term dependencies. 
                                            The network is composed of a series of layers, each of which contains a number of neurons. 
                                            The first layer is the input layer, which receives the input vectors. 
                                            The second layer is the hidden layer, which transforms the input vectors into a higher-dimensional space. 
                                            The third layer is the output layer, which transforms the hidden vectors into the output vectors.   
                                        """)
                            
                            model_feats = ("""
                                           <ul>
                                           <li><h5>Uses contextual information in mapping input and output sequences 
                                                <ul>
                                                    <li><p>Graves (2019)
                                                </ul>
                                            </li>
                                            <li><h5>Lcapable of learning longer temporal sequences, which enables the provision of better emotion classification accuracy
                                                <ul>
                                                    <li><p>Nakisa et al. (2018)
                                                </ul>
                                            </li>
                                            <li><h5>Performance significantly relies on the hyperparameter values
                                                <ul>
                                                     <li><p>Nakisa et al. (2018)
                                                </ul>
                                            </li>
                                            </ul>
                                           """)
                            
                            model_finds = ("""
                                          <ol>
                                            <li><h5>Captures emotion through the means of vectors which contain frame and sequence level information
                                                <ul>
                                                     <li><p>Chao et al. (2016)
                                                </ul>
                                            </li>
                                            <li><h5>Found success through its ability to handle the exploding/vanishing gradient problem.
                                                <ul>
                                                    <li><p>Van Houdt et al. (2020)
                                                </ul>
                                            </li>
                                            <li><h5>Training difficulties in the weights due to the lowering gradient aligning to the covered network distance.
                                                <ul>
                                                     <li><p>Van Houdt et al. (2020)
                                                </ul>
                                            </li>
                                          </ul>
                                        """)
                            
                        case "ResNet-50":
                            st.header("Residual Networks-50 (ResNet-50)")
                            model_desc = ("""
                                            ResNet-50 is a convolutional neural network that is 50 layers deep. 
                                            ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks. 
                                            The fundamental breakthrough with ResNet was it allowed us to train extremely deep neural networks with 150+ layers.
                                            It is an innovative neural network that was first introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun 
                                                in their 2015 computer vision research paper titled 'Deep Residual Learning for Image Recognition'.
                                        """)
                            
                            model_feats = ("""
                                           <ul>
                                           <li><h5>Capable of overcoming degradation problems using residual learning
                                                <ul>
                                                    <li><p>Celano (2021)
                                                </ul>
                                            </li>
                                            <li><h5>Performs exceptionally at audio classification tasks of longer duration with great stability
                                                <ul>
                                                    <li><p>Ayadi & Lachiri (2022)
                                                </ul>
                                            </li>
                                            <li><h5>Utilizes transfer learning and multiple convolutional layers 
                                                <ul>
                                                     <li><p>Pratama et al. (2021)
                                                </ul>
                                            </li>
                                            </ul>
                                           """)
                            
                            model_finds = ("""
                                          <ol>
                                            <li><h5>Model complexity & need for more training time proved to show the most accurate results
                                                <ul>
                                                     <li><p>Alnuaim et al. (2022)
                                                </ul>
                                            </li>
                                            <li><h5>Unable to generalize features from the training data without over or under-fitting
                                                <ul>
                                                     <li><p>Jain et al. (2021)
                                                </ul>
                                            </li>
                                          </ul>
                                        """)
                    
                    st.write(model_desc)
                    with st.expander("Model Information"):
                        st.subheader("FEATURES")
                        st.markdown(model_feats, unsafe_allow_html = True)
                        st.subheader("FINDINGS")                         
                        st.markdown(model_finds, unsafe_allow_html = True)
                    
                
        case "EXPERIMENTS":
            graph_desc = ''
            
            with st.container():
                columns = st.columns((.6,.2,2.5))
                with columns[0]:
                    sltd_model = st.selectbox("Select Model", ("Xception", "LSTM", "ResNet-50"))
                    
                with columns[2]:
                    
                    match sltd_model:
                        case "Xception":
                            #CHANGE SAMPLE CODE BELOW; conduct DBM()
                            # chart_data = pd.DataFrame(
                            #     np.random.randn(20, 3),
                            #     columns=['a', 'b', 'c'])
                            st.image("experiment/Xcept_train.png",f"{sltd_model} Accuracy and Loss.")
                            graph_desc = '''Xception, at the 100th epoch, 
                            resulted in a 96% training accuracy and 77% validation accuracy. 
                            The loss, on the other hand, has a training loss of 0.12 and a validation loss of 1.15. 
                            Looking at the graph, the model shows signs of overfitting at the 25th epoch, where the 
                            validation accuracy and loss starts to plateau. The best model saved was used to get the testing 
                            accuracy, which resulted in 76.8% accuracy.'''
                        
                        case "LSTM":
                            #CHANGE SAMPLE CODE BELOW; conduct  LSTM()
                            # chart_data = pd.DataFrame(
                            #     np.random.randn(20, 3),
                            #     columns=['a', 'b', 'c'])
                            st.image("experiment/LSTM_train.png",f"{sltd_model} Accuracy and Loss.")
                            graph_desc = """	LSTM scores, shows a training accuracy of 80% but only a validation accuracy of 62%. 
                            The validation loss of this model plateaus as early as the 40th epoch while the training loss keeps 
                            decreasing. As seen in the Figure above., both validation accuracy and loss keep diverging as training goes on
                            which is a sign of overfitting. The best saved model after training is used to take the training accuracy
                            which resulted in 62% also the same as the validation accuracy."""
                            
                        case "ResNet-50":
                            #CHANGE SAMPLE CODE BELOW; conduct ResNet-50()
                            model = ResNet50()

                            # chart_data = pd.DataFrame(
                            #     np.random.randn(20, 3),
                            #     columns=['a', 'b', 'c'])
                            # st.line_chart(chart_data)
                            st.image("experiment/res50_train.png",f"{sltd_model} Accuracy and Loss.")
                            graph_desc = """ResNet-50, after training, shows a training accuracy of 
                            81% and validation accuracy of 75%. The loss however, is disparate wherein 
                            the training loss is at 2.2 while the validation loss is at 26. As seen in the Figure above.
                             it is clearly seen that the training validation accuracy is still making progress 
                             per epoch which is a good sign that the model is learning well. The loss however 
                             tells the opposite, wherein the training loss keeps decreasing while the validation 
                             loss keeps fluctuating at much higher levels. Testing the best saved model showed a 
                             testing accuracy of 75%, the same as the validation accuracy."""
                    
                    with st.expander('Results Interpretation'):
                        st.write(graph_desc)
            
            with st.container():
                columns = st.columns((1,3))
                with columns[0]:
                    st.header('CONCLUSION')
                    
                with columns[1]:
                    st.write("""
                    The study concludes that there is a significant difference between the deep learning models. Xception performed the best at 77% accuracy, while 
                        ResNet-50 came close at 75% accuracy and LSTM performing at only 62% accuracy. Taking into account the results of how
                        well the models predicted certain emotion, the researchers also found a significant difference. Using one-way ANOVA, on precision and recall, the result showed significant
                        difference on precision but only showed significant difference on recall for happy, sad, angry and disgusted emotions.
                    """)
                    images = ["experiment/precision_s.png","experiment/recall_s.png"]
                    st.image(images,caption=["one-way ANOVA (Precision)","one-way ANOVA (Recall)"])
                # with columns[2]:
                #     st.image("experiment/recall_s.png","one-way ANOVA (Recall)")
                    
        case "MODEL TESTING":
            with st.container():
                col1, col2, col3 = st.columns([.5, .125, 2])
                col3.subheader('PROCESS LOG')
                
                with col1:
                    sltd_model = st.selectbox("Select Model", ("Xception", "LSTM", "ResNet-50"))
                    # sltd_feat = st.selectbox("Extraction Feature", ("LPC", "LPCC", "MFCC  "))
                    epochs_in = st.number_input("No. of Epochs", value=2, format='%i',min_value=2,step=1)
                    X,Y = config.data

                    X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=42)

                    if st.button("TEST!"):
                        #modiffy code below, ma dud
                        with col3:
                            with st.spinner('Running model test w/ chosen no. of epochs...'):

                                #INSERT EXTRA/MODIFY CODE HERE, MA DUD
                                match sltd_model:
                                    case "Xception":
                                        #CHANGE CODE HERE, conduct DBM()
                                        input = Input(shape = (9,13,1))
                                        x = entry_flow(input)
                                        x = middle_flow(x)
                                        output = exit_flow(x)

                                        model = Model (inputs=input, outputs=output)
                                        model.summary()
                                        model.compile(optimizer='adam',
                                                        loss='categorical_crossentropy',
                                                        metrics=['accuracy'])
                                        # pyrefly: ignore [unexpected-keyword]
                                        checkpoint = tf.keras.callbacks.ModelCheckpoint("models/Xceptest.model",monitor="val_accuracy",verbose=1,mode="max",save_best_only=True,save_weights_only=False,period=1)
                                        history = model.fit(X_train,y_train,epochs=epochs_in,batch_size=32,
                                                shuffle=True,validation_split=0.1,callbacks=[CustomCallback()])                                                
                                        y_pred = model.predict(X_test)

                                        model_acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))
                                        class_acc,class_prc = metric_df(y_test,y_pred)
                                        st.write(f"Model accuracy: {round(model_acc,4)}")
                                        st.pyplot(show_plot(history.history))
                                        st.write(class_acc)
                                        st.write(class_prc)
                                    case "LSTM":
                                        model = lstm()
                                        X2 = X.reshape(X.shape[0],X.shape[1],X.shape[2])

                                        X_train,X_test, y_train,y_test = train_test_split(X2,Y,test_size=0.1,random_state=42)

                                        # pyrefly: ignore [unexpected-keyword]
                                        checkpoint = tf.keras.callbacks.ModelCheckpoint("models/lstest.model",monitor="val_accuracy",verbose=1,mode="max",save_best_only=True,save_weights_only=False,period=1)
                                        history = model.fit(X_train,y_train,epochs=epochs_in,batch_size=32,
                                                shuffle=True,validation_split=0.1,callbacks=[CustomCallback()])
                                        y_pred = model.predict(X_test)

                                        model_acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))
                                        class_acc,class_prc = metric_df(y_test,y_pred)
                                        st.write(f"Model accuracy: {round(model_acc,4)}")
                                        st.pyplot(show_plot(history.history))
                                        st.write(class_acc)
                                        st.write(class_prc)
                                        # for x in range(epochs_in):
                                        #     st.write('placeholder for LSTM processing')
                                        #     time.sleep(1)
                                    case "ResNet-50":
                                        model = ResNet50()
                                        model.compile(loss="categorical_crossentropy",
                                                    optimizer="adam",
                                                    metrics=["accuracy"])
                                        # pyrefly: ignore [unexpected-keyword]
                                        checkpoint = tf.keras.callbacks.ModelCheckpoint("models/restest.model",monitor="val_accuracy",verbose=1,mode="max",save_best_only=True,save_weights_only=False,period=1)
                                        history = model.fit(X_train,y_train,epochs=epochs_in,batch_size=32,
                                                shuffle=True,validation_split=0.1,callbacks=[CustomCallback()])
                                        y_pred = model.predict(X_test)

                                        model_acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))
                                        class_acc,class_prc = metric_df(y_test,y_pred)
                                        st.write(f"Model accuracy: {round(model_acc,4)}")
                                        st.pyplot(show_plot(history.history))
                                        st.write(class_acc)
                                        st.write("\n\n\n\n"+class_prc)
                                        # for x in range(epochs_in):
                                        #     st.write('placeholder for ResNet-50 processing')
                                        #     time.sleep(1)
                                
                                st.success('Done!') 
                                
        case "SYSTEM DEMO": 
            st.markdown('''
                        <h2 align=center>ABOUT THE DETECTOR</h2>
                        
                        <p style="text-align:center;">
                            The developers/researchers present a tool which embodies the culmination of their experiments.
                            The Speech Emotion Detector is a tool that can be used to automatically detect the emotions conveyed in a speech signal. 
                            This tool can be used in a variety of settings, from personal relationships to professional ones.                             
                            The detector uses a combination of acoustic features and machine learning algorithms to classify the emotions in a speech signal. 
                            The emotions that can be detected include the following: neutrality, calmness, happiness, sadness, anger, fear, and disgust.
                        </p>
                        ''', unsafe_allow_html = True)
                
            with st.container():
                columns = st.columns((1, .5, 1.5, .5, 1))
                uploaded_file = None
                sltd_model = None
                with columns[2]:                    
                    sltd_model = st.selectbox("Select Model", ("Xception", "LSTM", "ResNet-50"))
                
                with st.container():
                    options = st.columns((.5,1.5,1.5,.5))
                    with options[1]:
                        st.caption("Record Audio")
                        val = st_audiorec()
                        if isinstance(val, dict):  # retrieve audio data
                            with st.spinner('retrieving audio-recording...'):
                                ind, val = zip(*val['arr'].items())
                                ind = np.array(ind, dtype=int)  # convert to np array
                                val = np.array(val)             # convert to np array
                                sorted_ints = val[ind]
                                stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
                                wav_bytes = stream.read()
                            
                                # get the format of the audio as wav or mp3
                                with open("filename.wav", "wb") as audio_file:
                                    audio_file.write(wav_bytes)
                                uploaded_file = 1
                                
                    with options[2]:
                        #snippet for uploaded WAV file as variable
                        if uploaded_file != 1:
                            uploaded_file = st.file_uploader("Upload Audio", accept_multiple_files = False, type = 'wav')
                        if uploaded_file is not None and uploaded_file != 1:
                            st.audio(uploaded_file)
                        
                        classify_btn = st.button('CLASSIFY')
                    
                    columns = st.columns((.5, 4, .5))
                    with columns[1]:
                        if classify_btn:    
                            class_dict,fn_prob,signals,pred_s = classify(uploaded_file,sltd_model)         
                            showResults(class_dict, fn_prob, signals, pred_s)


              
if __name__ == '__main__':
    main()