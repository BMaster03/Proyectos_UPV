import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential #type: ignore
from keras.layers import Dense, Input #type: ignore 
from keras.utils import to_categorical #type: ignore
from keras.datasets import mnist #type: ignore

# Training data 
(train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data() # Data: input data, and labels: used to calculate the loss function
#mnist.load_data() esto es un método, descarga las imagnes de internt, todo esto es una tupla

# Information about the training data
print(train_data_x.shape) #que forma tiene el vector de entrenamineto, un tensor con 60mil imagenes de 28*28
print(train_labels_y[1]) #que etiqueta tiene la primera imagen 
print(test_data_x.shape)
plt.imshow(train_data_x[1])

# Neural network architecture using TensorFlow and Keras
model = Sequential([ 
    Input(shape = (28*28,)), #como un vector
    Dense(512, activation = 'relu'), # number of neurons
    Dense(10, activation = 'softmax')
    ])

# Compile the model 
model.compile( 
    optimizer = 'rmsprop', #función de pérdida, es la forma que actualizamos los pesos, basado en el alg. del gradienteq
    loss = 'categorical_crossentropy', #type of loss function 
    metrics = ['accuracy']
    )

# Resume of model 
model.summary() #mostrar la red que se está construyendo

# Training data preprocessing, generate of normality of data
x_train = train_data_x.reshape(60000, 28*28) #converitmos a una matriz, un tensor de 3d a 2d
x_train = x_train.astype('float32')/255 # generate of normality, generamos los blancos y negros, 0 y 1
y_train = to_categorical(train_labels_y)

# Testing data preprocessing
x_test = test_data_x.reshape(10000, 28*28)
x_test = x_test.astype('float32')/255 # generate of normality  
y_test = to_categorical(test_labels_y)

# Train the model 
model.fit(x_train, y_train, epochs = 3, batch_size = 128) # generate the train five, or five cicles, and the batch is for how pase the imagenes
#pasamos en 128 en 128 imagenes en paquetes, en 3 ciclos. 

# Evaluate of model, the Neural Network 
model.evaluate(x_test,y_test) #evaluate of model utilice date of test 



