import flwr as fl
import tensorflow as tf
from pickle import load
import numpy as np
from sys import argv
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import random 

serverPort = '8080'

if len(argv) >= 2:
    serverPort = argv[1]

# first class
x_train1 = np.asarray(load(open('datasets/CIFAR-10/Non-IID-distribution/train/class0Train','rb')))
y_train1 = np.asarray(load(open('datasets/CIFAR-10/Non-IID-distribution/train/class0TrainLabel','rb')))
x_test1 = np.asarray(load(open('datasets/CIFAR-10/Non-IID-distribution/test/class0Test','rb')))
y_test1 = np.asarray(load(open('datasets/CIFAR-10/Non-IID-distribution/test/class0TestLabel','rb')))

x_train1 = x_train1[1250:2500]
y_train1 = y_train1[1250:2500]
x_test1 = x_test1[250:500]
y_test1 = y_test1[250:500]


# second class
x_train2 = np.asarray(load(open('datasets/CIFAR-10/Non-IID-distribution/train/class1Train','rb')))
y_train2 = np.asarray(load(open('datasets/CIFAR-10/Non-IID-distribution/train/class1TrainLabel','rb')))
x_test2 = np.asarray(load(open('datasets/CIFAR-10/Non-IID-distribution/test/class1Test','rb')))
y_test2 = np.asarray(load(open('datasets/CIFAR-10/Non-IID-distribution/test/class1TestLabel','rb')))

x_train2 = x_train2[1250:2500]
y_train2 = y_train2[1250:2500]
x_test2 = x_test2[250:500]
y_test2 = y_test2[250:500]

# create the dataset

X = np.concatenate((x_train1,x_train2,x_test1,x_test2))
Y = np.concatenate((y_train1,y_train2,y_test1,y_test2))

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.65, random_state=47527)
train_index, test_index = next(sss.split(X, Y))

x_train = X[train_index]
x_test = X[test_index]
y_train = Y[train_index]
y_test = Y[test_index]

model = keras.Sequential()

# Model adapted from https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa

# Convolutional layer and maxpool layer 1
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3),padding='same'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 2
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 3
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 4
model.add(keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
model.add(keras.layers.Flatten())

# Hidden layer with 512 neurons and Rectified Linear Unit activation function 
model.add(keras.layers.Dense(512,activation='relu'))

#Here we use sigmoid activation function which makes our model output to lie between 0 and 1
model.add(keras.layers.Dense(10,activation='sigmoid'))

# compile the model
model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics='accuracy')


def numero_classes(y):
    zero = 0
    um = 0
    for x in y:
        if(x[0]==1):
            um+=1
        else:
            zero+=1        
    print('Fit Classe 1:', zero, '  Classe 2:', um)
    return (zero,um)

def calcular_numero_amostras(n_dados_classe_1, n_dados_classe_2, nivel_confianca, erro_maximo):
    p1 = (n_dados_classe_1/(n_dados_classe_1 + n_dados_classe_2))
    desvio_padrao = np.sqrt(p1*(1-p1))
    N = n_dados_classe_1+n_dados_classe_2
    Z = nivel_confianca
    B = erro_maximo
    numerador = N*(Z**2)*(desvio_padrao**2)
    denominador = (B**2)*(N-1)+(Z**2)*(desvio_padrao**2)
    numero_amostras = numerador/denominador
    return round(numero_amostras)


# federated client
class CifarClient(fl.client.NumPyClient):
	
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        x = random.randint(0, 10000) #ate 2100
        x_train_s, y_train_s = shuffle(x_train, y_train, random_state=x)
        (n_dados_classe_1, n_dados_classe_2) = numero_classes(y_train_s)
        numero_amostras = calcular_numero_amostras(n_dados_classe_1, n_dados_classe_2, 1.96, 0.05)

        model.fit(x_train_s[0:numero_amostras],y_train_s[0:numero_amostras],validation_split=0.2,batch_size=70,epochs=5,verbose=1)
        return model.get_weights(), len(x_train_s[0:numero_amostras]), {}


    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
        return loss, len(x_test), {"accuracy": accuracy}
       
fl.client.start_numpy_client(server_address='localhost:'+serverPort, client=CifarClient())

