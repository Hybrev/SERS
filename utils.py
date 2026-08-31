import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pandas as pd
import os
import pickle
from experiment.cfg import Config
from sklearn.metrics import classification_report,accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv2D,Add
from tensorflow.keras.layers import SeparableConv2D,ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool2D,LSTM,Dropout
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model


from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import kapre
from kapre.composed import get_melspectrogram_layer
import os
from python_speech_features import mfcc
import pandas as pd
import numpy as np

import streamlit as st
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from keras.models import Sequential
from keras.models import Model

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from scipy.io import wavfile
from keras.utils import to_categorical
import pickle
import librosa

from sklearn.model_selection import train_test_split
data_dir = "../../../RAVDESS"
save_dir = "../../dataset/"
clean_dir = save_dir+"clean/"

df = pd.read_csv("experiment/audio_data.csv",index_col=[0])
classes = list(df["emotion"].unique())

    

def identity_block(x, f,filters):
    # copy tensor to variable called x_skip
    x_skip = x
    F1,F2,F3 = filters
    # Layer 1
    x = tf.keras.layers.Conv2D(F1, (1,1), padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(F2, (f,f), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)     
    x = tf.keras.layers.Activation('relu')(x)
    
    #layer 3
    x = tf.keras.layers.Conv2D(F3, (1,1), padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    #add residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def convolutional_block(x, f,filters,s=2):
    # copy tensor to variable called x_skip
    x_skip = x
    F1,F2,F3 = filters
    
    #layer 1
    x = tf.keras.layers.Conv2D(F1, (1,1), padding = 'valid', strides = (s,s))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(F2, (f,f), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 3
    x = tf.keras.layers.Conv2D(F3, (1,1), padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(F3, (1,1), strides = (s,s),padding="valid")(x_skip)
    x_skip = tf.keras.layers.BatchNormalization(axis=3)(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNet50(shape = (9,13,1), classes = 8):
    
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
    filter_size = 64
    #1
    x = convolutional_block(x,3,[64,64,256],s=1)
    x = identity_block(x,3,[64,64,256])
    x = identity_block(x,3,[64,64,256])
    #2
    x= convolutional_block(x,3,[128,128,512],s=2)
    x= identity_block(x,3,[128,128,512])
    x= identity_block(x,3,[128,128,512])
    x= identity_block(x,3,[128,128,512])
    #3
    x= convolutional_block(x,3,[256,256,1024],s=2)
    x= identity_block(x,3,[256,256,1024])
    x= identity_block(x,3,[256,256,1024])
    x= identity_block(x,3,[256,256,1024])
    x= identity_block(x,3,[256,256,1024])
    x= identity_block(x,3,[256,256,1024])
    #4
    x= convolutional_block(x,3,[512,512,2056],s=2)
    x= identity_block(x,3,[512,512,2056])
    x= identity_block(x,3,[512,512,2056])

    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet50")
    return model



#XCEPTION
# creating the Conv-Batch Norm block
def conv_bn(x, filters, kernel_size, strides=1):
    
    x = Conv2D(filters=filters, 
               kernel_size = kernel_size, 
               strides=strides, 
               padding = 'same', 
               use_bias = False)(x)
    x = BatchNormalization()(x)
    return x

# creating separableConv-Batch Norm block

def sep_bn(x, filters, kernel_size, strides=1):
    
    x = SeparableConv2D(filters=filters, 
                        kernel_size = kernel_size, 
                        strides=strides, 
                        padding = 'same', 
                        use_bias = False)(x)
    x = BatchNormalization()(x)
    return x
# entry flow

def entry_flow(x):
    
    x = conv_bn(x, filters =32, kernel_size =3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters =64, kernel_size =3, strides=1)
    tensor = ReLU()(x)
    
    x = sep_bn(tensor, filters = 128, kernel_size =3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 128, kernel_size =3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=128, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    
    x = ReLU()(x)
    x = sep_bn(x, filters =256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters =256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=256, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    
    x = ReLU()(x)
    x = sep_bn(x, filters =728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters =728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=728, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    return x
# middle flow

def middle_flow(tensor):
    
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        tensor = Add()([tensor,x])
        
    return tensor

# exit flow

def exit_flow(tensor):
    
    x = ReLU()(tensor)
    x = sep_bn(x, filters = 728,  kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 1024,  kernel_size=3)
    x = MaxPool2D(pool_size = 3, strides = 2, padding ='same')(x)
    
    tensor = conv_bn(tensor, filters =1024, kernel_size=1, strides =2)
    x = Add()([tensor,x])
    
    x = sep_bn(x, filters = 1536,  kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 2048,  kernel_size=3)
    x = GlobalAvgPool2D()(x)
    
    x = Dense (units = 8, activation = 'softmax')(x)
    
    return x
def show_plot(history):
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(20,5))

    ax1.plot(history['accuracy'],**{"marker":"o"})
    ax1.plot(history['val_accuracy'],**{"marker":"o"})
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'val'], loc='upper left')

    ax2.plot(history['loss'],**{"marker":"o"})
    ax2.plot(history['val_loss'],**{"marker":"o"})
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'val'], loc='upper left')
    return fig

def lstm():
    #shape of RNN is (n,time,feat)
    model = Sequential()
    model.add(LSTM(128,return_sequences=True,input_shape=(9,13)))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64,activation="relu")))
    model.add(TimeDistributed(Dense(32,activation="relu")))
    model.add(TimeDistributed(Dense(16,activation="relu")))
    model.add(TimeDistributed(Dense(8,activation="relu")))
    model.add(Flatten())
    model.add(Dense(8,activation="softmax"))
    print(model.summary())
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                 metrics=["accuracy"])
    return model



def class_report(df,label):
    y_true = []
    y_pred = []
    for x in df.iterrows():
        if x[1]["true_emotion"] == label:
            y_true.append(True)
        else:
            y_true.append(False)

        if x[1]["pred_emotion"] == label:
            y_pred.append(True)
        else:
            y_pred.append(False)
    return y_true,y_pred

def class_classification_score(df):
    accuracy_by_class = {"emotion":[],"accuracy":[]}
    for emotion in classes:
        y_true,y_pred = class_report(df,emotion)
        accuracy = accuracy_score(y_true,y_pred)
        accuracy_by_class["emotion"].append(emotion)
        accuracy_by_class["accuracy"].append(accuracy)
    return pd.DataFrame(accuracy_by_class),pd.DataFrame(classification_report(df["true_emotion"],df["pred_emotion"],output_dict=True)).T
def metric_df(y_test,y_pred):
    y_pr = np.argmax(y_pred,axis=1)
    y_tr = np.argmax(y_test,axis=1)
    y_pproba = [x[y] for x,y in zip(y_pred,y_pr)]
    df = pd.DataFrame({"y_true":y_tr,"y_pred":y_pr})
    df["true_emotion"] = df["y_true"].apply(lambda x: classes[x])
    df["pred_emotion"] = df["y_pred"].apply(lambda x: classes[x])
    return class_classification_score(df)


def plot_signals(signals,pred_s):
    Tot = len(signals)
    Cols = 5

    # Compute Rows required

    Rows = Tot // Cols 
    Rows += Tot % Cols

    # Create a Position index

    Position = range(1,Tot + 1)

    # Create main figure
    i = 0
    fig = plt.figure(figsize=(15,10))
    title_color = {"neutral":"gray","calm":"cyan","happy":"yellow","sad":"blue","angry":"red","fearful":"purple","disgust":"green","surprised":"orange"}
    fig.suptitle("Total number of segments: "+str(len(signals)))
    for k in range(Tot):

      # add every single subplot to the figure with a for loop
        label = pred_s[i]
        ax = fig.add_subplot(Rows,Cols,Position[k])
        ax.set_title(label,color=title_color[label])
        ax.plot(list(signals)[i])      # Or whatever you want in the subplot
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        i += 1
    fig.tight_layout()
    return fig


class CustomCallback(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        st.write(f"Epoch {epoch+1}: model is training...")

    def on_epoch_end(self, epoch, logs=None):
        st.write(f"End of epoch {epoch+1} - loss: {round(logs['loss'],4)} - accuracy: {round(logs['accuracy'],4)} - val_loss: {round(logs['val_loss'],4)} - val_acc: {round(logs['val_accuracy'],4)}")
