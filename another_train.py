# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:17:10 2021

@author: 57821
"""

import numpy as np
import mnist
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from matplotlib import pyplot
import pandas



# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('bottle/train', class_mode='binary',target_size=(28,28),color_mode='rgb')
test_it = datagen.flow_from_directory('bottle/test', class_mode='binary',target_size=(28,28),color_mode='rgb')
# confirm the iterator works
train_images, trainlabel = train_it.next()
test_images, testlabel = test_it.next()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5


print(train_images.shape) 
print(test_images.shape)

# Build the model.
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),padding='same',
                 activation='relu',
                 input_shape=(28,28,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Compile the model.
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
history = model.fit(train_images, to_categorical(trainlabel), 
            validation_data =(test_images, to_categorical(testlabel)), epochs=40, batch_size=30, verbose=2)

# Evaluate the model.
model.evaluate(
  test_images,
  to_categorical(testlabel)
)


print(history.history.keys())


def get_curves(history):
    if 'accuracy' in history.history.keys():
        pyplot.plot(history.history['accuracy'])
    else:
        pyplot.plot(history.history['acc'])
#    pyplot.plot(history.history['val_accuracy'])
    if 'val_accuracy' in history.history.keys():
        pyplot.plot(history.history['val_accuracy'])
    else:
        pyplot.plot(history.history['val_acc'])
#    pyplot.plot(history.history['val_accuracy'])
    pyplot.title('model accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'test'], loc='upper left')
    pyplot.savefig('accuracy_history.png')
    pyplot.show()
    pyplot.close()
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'test'], loc='upper left')
    pyplot.savefig('loss_history.png')
    pyplot.show()
    pyplot.close()

#saving the weights
model.save_weights('model.h5')
#saving the graph representation of network structure

predictions = model.predict(test_images[:])
print("predictions of the bottles' label")
print(np.argmax(predictions, axis=1)) 
testlabel = testlabel.astype(int)  
print("the test lable")
print(testlabel[:]) 
get_curves(history)
hist = pandas.DataFrame(history.history) 
history_file = 'history.csv'
with open(history_file, mode='w') as f:
	hist.to_csv(f)