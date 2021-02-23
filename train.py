from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from matplotlib import pyplot
import pandas

# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('bottle/train', class_mode='binary',target_size=(28,28),color_mode='grayscale')
test_it = datagen.flow_from_directory('bottle/test', class_mode='binary',target_size=(28,28),color_mode='grayscale')
# confirm the iterator works
trainimage, trainlabel = train_it.next()
testimage, testlabel = test_it.next()

# normalize the pixel values from [0, 255] to [-0.5, 0.5] to make our network easier to train.
trainimage = (trainimage/ 255) - 0.5
testimage = (testimage / 255) - 0.5

# Flatten the images.
trainimage = trainimage.reshape((-1, 784))
testimage = testimage.reshape((-1, 784))

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(5, activation='softmax'),
])

# Compile the model.
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
#  loss='binary_crossentropy',
  metrics=['accuracy'],
)

# Train the model.

print('hello')
'''
history=model.fit(
  trainimage,
  to_categorical(trainlabel),
  epochs=20,
  batch_size=32,
)
'''
history = model.fit(trainimage, to_categorical(trainlabel), \
            validation_data =(testimage, to_categorical(testlabel)), epochs=40, batch_size=30, verbose=2)

# Evaluate the model.
model.evaluate(
  testimage,
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

predictions = model.predict(testimage[:])
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

