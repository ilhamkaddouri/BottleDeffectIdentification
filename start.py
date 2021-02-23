from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# define location of dataset
folder = 'bottles/'
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
    
	# determine class
    output = 0.0
    if file.startswith('nolabel'):
        output = 1.0
    if file.startswith('normal'):
        output=2.0
    if file.startswith('overfilled'):
        output=3.0
    if file.startswith('underfilled'):
        output=10.0

	# load image
	# convert to numpy array
    photo = load_img(folder + file, target_size=(20, 20))
    photo = img_to_array(photo)
	# store
    photos.append(photo)
    labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('cola_photos.npy', photos)
save('cola_labels.npy', labels)