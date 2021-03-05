import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math

#1
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#2
def display_stats():
    plt.hist(y_train,rwidth=0.9, bins=10)
    plt.show()
    
#display_stats()

#3
def display_img(img):
    imgplot = plt.imshow(img)
    plt.show()
    
#display_img(x_train[1])

#4
def display_each_img(x_train,y_train):
    print(y_train)
    compteur=0
    y_train = list(y_train)
    
    for y in y_train:
        if y == compteur:
            compteur+=1
            display_img(x_train[y_train.index(y)])
        elif compteur == 10:
            return
            
        
display_each_img(x_train,y_train)
