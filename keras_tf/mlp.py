
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import random
import time

#print(tf.config.list_physical_devices('GPU'))


# Fijar la semilla
semilla = 42

random.seed(semilla) # Python
np.random.seed(semilla) # NumPy y Pandas
tf.random.set_seed(semilla) # TensorFlow


# Cargar el conjunto de entrenamiento
datos = pd.read_csv('datasets/fashion-mnist/fashion-mnist_train.csv')

# Extraer etiquetas y píxeles
etiquetas = datos['label']
imagenes = datos.drop('label', axis=1).values


# Escalar las imágenes entre 0 y 1
imagenes = imagenes / 255.0


# Dividir en entrenamiento y validación (80%-20%)
trainX, validX, trainY, validY = train_test_split(imagenes, etiquetas, test_size = 0.2, stratify = etiquetas, random_state = semilla)


# Cargar el conjunto de prueba
test = pd.read_csv('fashion-mnist/fashion-mnist_test.csv')

# Extraer etiquetas y píxeles
testY = test['label']
testX = test.drop('label', axis=1).values / 255.0


# Crear el modelo con la configuración 1

def crearModelo1():
    funcActivacion = "relu"
    dropout = 0.2
    lr = 0.001
    
    modelo = Sequential([
        # Input + Capa oculta 1
        Dense(256, activation=funcActivacion, input_shape=(trainX.shape[1],)),
        Dropout(dropout),
        
        # Capa oculta 2
        Dense(128, activation=funcActivacion),
        Dropout(dropout),
        
        # Capa de salida
        Dense(10, activation='softmax')
    ])
    
    # Compilar el modelo
    modelo.compile(
        optimizer = Adam(learning_rate = lr),             
        loss = 'sparse_categorical_crossentropy',  # Loss para clases mutuamente excluyentes
        metrics = ['accuracy']
    )
    
    # Mostrar la estructura del modelo
    modelo.summary()

    return modelo



# Crear el modelo con la configuración 1

def crearModelo2():
    funcActivacion = "relu"
    dropout = 0.2
    lr = 0.001
    
    modelo = Sequential([
        # Input + Capa oculta 1
        Dense(512, activation=funcActivacion, input_shape=(trainX.shape[1],)),
        Dropout(dropout),
        
        # Capa oculta 2
        Dense(512, activation=funcActivacion),
        Dropout(dropout),
        
        # Capa oculta 2
        Dense(256, activation=funcActivacion),
        Dropout(dropout),
        
        # Capa oculta 2
        Dense(256, activation=funcActivacion),
        Dropout(dropout),
        
        # Capa oculta 2
        Dense(128, activation=funcActivacion),
        Dropout(dropout),
        
        # Capa de salida
        Dense(10, activation='softmax')
    ])
    
    # Compilar el modelo
    modelo.compile(
        optimizer = Adam(learning_rate = lr),             
        loss = 'sparse_categorical_crossentropy',  # Loss para clases mutuamente excluyentes
        metrics = ['accuracy']
    )
    
    # Mostrar la estructura del modelo
    modelo.summary()

    return modelo



modelo1 = crearModelo1()


# Entrenamiento midiendo el tiempo
inicio = time.time()

history = modelo1.fit(trainX, trainY, epochs = 30, validation_data = (validX, validY))

fin = time.time()

# Calcular el tiempo transcurrido
tiempo = fin - inicio
print(f"Tiempo de entrenamiento: {tiempo:.2f} segundos")


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


modelo1.evaluate(testX, testY)


modelo2 = crearModelo2()


# Entrenamiento midiendo el tiempo
inicio = time.time()

history = modelo2.fit(trainX, trainY, epochs = 30, validation_data = (validX, validY))

fin = time.time()

# Calcular el tiempo transcurrido
tiempo = fin - inicio
print(f"Tiempo de entrenamiento: {tiempo:.2f} segundos")


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


modelo2.evaluate(testX, testY)





