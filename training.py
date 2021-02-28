#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


import tensorflow.keras
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.layers as L
import zipfile
import pandas as pd
import io
import random
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras import losses
import json
import tensorflow as tf
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import time
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from skimage.transform import resize

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

res = np.load("complexs.npy", allow_pickle=True)
for i in range(len(res)):
    res[i] = np.transpose(res[i],(0,2,1))

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]

    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_kl(ignore, kl):
    return K.mean(kl)

def lossf(y_true,y_pred):
    return (get_rc(y_pred, y_true) + get_kl(y_pred, y_true))

def get_rc(y_true,y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true)))

autoencoder = load_model('autoencoder3.h5', custom_objects={'get_rc': get_rc, 'get_kl': get_kl})
encoder = load_model("encoder3.h5")
decoder = load_model("decoder3.h5")
autoencoder.compile(optimizer='adam', loss=[get_rc, get_kl], loss_weights = [1, 0.35])

def generator_model(batch_size):
    while True:
      answ = np.zeros((batch_size, 400, 1))
      i = 0                  
      while i < batch_size:
        index = np.random.choice(len(res)) 
        numb = np.random.choice(len(res[index])) 
        lead = np.random.choice(12)
        ecg = res[index][numb,:, lead]
      
        answ[i,:,0] = ecg
        i += 1
      yield (answ,[answ, answ])

gener = generator_model(64)

while True:
    history = autoencoder.fit_generator(gener, epochs=30, steps_per_epoch=64)
    autoencoder.save('autoencoder3.h5')
    decoder.save('decoder3.h5')
    encoder.save('encoder3.h5')

def generate(decoder, n_examples):
    results = decoder.predict(np.random.randn(n_examples, 25))
    for i in range(n_examples):	
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(results[i])
        #ax.axis('off')
        #buf = io.BytesIO()
        #buf = resize(buf, (350, 350), mode='reflect')
        #ax.tight_layout()
        fig.savefig("examples/exper_2lead11"+str(i)+".png", dpi=75)
        fig.clf()
        #fig.close()

generate(decoder, 10)

def generate25(encoder, n_examples):
    results = encoder.predict(np.random.randn(n_examples, 400))
    for i in range(n_examples):	
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(results[2][i])
        #ax.axis('off')
        #buf = io.BytesIO()
        #buf = resize(buf, (350, 350), mode='reflect')
        #ax.tight_layout()
        fig.savefig("examples/exper_encoder_"+str(i)+".png", dpi=75)
        fig.clf()
        #fig.close()

generate25(encoder, 100)
