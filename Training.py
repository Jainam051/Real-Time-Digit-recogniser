import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense , Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras import regularizers
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint
import h5py
import os
import cv2
from sklearn.model_selection import train_test_split


def data_loader(path_train,path_test):
   train_list=os.listdir(path_train)
   
   num_classes=len(train_list)

  
   x_train=[]
   y_train=[]
   x_test=[]
   y_test=[]

   # Loading training dataset
   for label,elem in enumerate(train_list):

           path1=path_train+'/'+str(elem)
           images=os.listdir(path1)
           for elem2 in images:
               path2=path1+'/'+str(elem2)
               
               img = cv2.imread(path2)
              
               img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
               img=img.reshape(28,28,1)
              
               x_train.append(img)
               
               y_train.append(str(label))

           
           path1=path_test+'/'+str(elem)
           images=os.listdir(path1)
           for elem2 in images:
               path2=path1+'/'+str(elem2)
              
               img = cv2.imread(path2)
               
               img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
               img=img.reshape(28,28,1)
              
              
               x_test.append(img)
              
               y_test.append(str(label))

   
   x_train=np.asarray(x_train)
   y_train=np.asarray(y_train)
   x_test=np.asarray(x_test)
   y_test=np.asarray(y_test)
   return x_train,y_train,x_test,y_test


path_train='./Data/train'
path_test='./Data/test'

X_train,y_train,X_test,y_test=data_loader(path_train,path_test)

input_shape = (X_train.shape[1], X_train.shape[2],1 )
print(X_train.shape)

#converting precision of pixels to 32-bit
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalization
X_train = X_train / 255.
X_test = X_test / 255.

#One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def model():
	
	model = Sequential()
	
	model.add(Conv2D(32, (3, 3), strides=(1, 1), padding = 'same' , input_shape = input_shape, activation = 'relu', kernel_initializer = 'glorot_uniform', kernel_regularizer = regularizers.l2(0.01)))
	model.add(Conv2D(32, (3, 3), strides=(1, 1), padding = 'same', activation = 'relu', kernel_initializer = 'glorot_uniform', kernel_regularizer = regularizers.l2(0.01)))
	
	model.add(MaxPooling2D((2, 2), strides=(2, 2), padding = 'valid'))
	
	model.add(Conv2D(64, (3, 3), strides=(1, 1), padding = 'same', activation = 'relu', kernel_initializer = 'glorot_uniform', kernel_regularizer = regularizers.l2(0.01)))
	model.add(Conv2D(64, (3, 3), strides=(1, 1), padding = 'same', activation = 'relu', kernel_initializer = 'glorot_uniform', kernel_regularizer = regularizers.l2(0.01)))
	
	model.add(Flatten())
	
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.3))

	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	return model

model = model()

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=200, verbose=1)

model.save('trained_model.h5')

scores = model.evaluate(X_test, y_test, verbose=1)
print(scores)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
