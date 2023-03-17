import os
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from random import shuffle
import random

from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

SIZE = 50
EPOCHS = 10
BATCH_SIZE = 128

PATH = 'C:/Users/Maedr3/Documents/packt/chapter 7/data/dogs-vs-cats/train/'

def get_input(file):
    return Image.open(PATH+file)

def get_output(file):
    class_label = file.split('.')[0]
    if class_label == 'dog': label_vector = [1,0]
    elif class_label == 'cat': label_vector = [0,1]
    return label_vector

def random_horizontal_flip(image):
    toss = random.randint(1, 2)
    if toss == 1:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return image

def random_rotate(image, rot_range):
    value = random.randint(-rot_range,rot_range)
    return image.rotate(value)

def random_horizontal_shift(image, shift):
    width, height = image.size
    rand_shift = random.randint(0,shift*width)
    image = PIL.ImageChops.offset(image, rand_shift, 0)
    image.paste((0), (0, 0, rand_shift, height))
    return image

def random_vertical_shift(image, shift):
    width, height = image.size
    rand_shift = random.randint(0,shift*height)
    image = PIL.ImageChops.offset(image, 0, rand_shift)
    image.paste((0), (0, 0, width, rand_shift))
    return image

def preprocess_input(image):
    
    # Data preprocessing
    image = image.convert('L')
    image = image.resize((SIZE,SIZE))
    
    
    # Data augmentation
    random_vertical_shift(image, shift=0.2)
    random_horizontal_shift(image, shift=0.2)
    random_rotate(image, rot_range=45)
    random_horizontal_flip(image)
    
    return np.array(image).reshape(SIZE,SIZE,1)

def custom_image_generator(images, batch_size = 128):
    
    while True:
        # Randomly select images for the batch
        batch_images = np.random.choice(images, size = batch_size)
        batch_input = []
        batch_output = [] 
        
        # Read image, perform preprocessing and get labels
        for file in batch_images:
            # Function that reads and returns the image
            input_image = get_input(file)
            # Function that gets the label of the image
            label = get_output(file)
            # Function that pre-processes and augments the image
            image = preprocess_input(input_image)

            batch_input.append(image)
            batch_output.append(label)

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        # Return a tuple of (images,labels) to feed the network
        yield(batch_x, batch_y)

def get_label(file):
    class_label = file.split('.')[0]
    if class_label == 'dog': label_vector = [1,0]
    elif class_label == 'cat': label_vector = [0,1]
    return label_vector

def get_data(files):
    data_image = []
    labels = []
    for image in tqdm(files):
        
        label_vector = get_label(image)
        

        img = Image.open(PATH + image).convert('L')
        img = img.resize((SIZE,SIZE))
        
       
        labels.append(label_vector)
        data_image.append(np.asarray(img).reshape(SIZE,SIZE,1))
        
    data_x = np.array(data_image)
    data_y = np.array(labels)
        
    return (data_x, data_y)

files = os.listdir(PATH)

random.shuffle(files)

train = files[:7000]
test = files[7000:]

validation_data = get_data(test)

model = Sequential()
    
model.add(Conv2D(48, (3, 3), activation='relu', padding='same', input_shape=(50,50,1)))    
model.add(Conv2D(48, (3, 3), activation='relu'))    
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.10))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics = ['accuracy'])

model_details = model.fit_generator(custom_image_generator(train, batch_size = BATCH_SIZE),
                    steps_per_epoch = len(train) // BATCH_SIZE, 
                    epochs = EPOCHS, 
                    validation_data= validation_data,
                    verbose=1)

score = model.evaluate(validation_data[0], validation_data[1])
print("Accuracy: {0:.2f}%".format(score[1]*100))

Accuracy: 72.57%

y_pred = model.predict(validation_data[0])

correct_indices = np.nonzero(np.argmax(y_pred,axis=1) == np.argmax(validation_data[1],axis=1))[0]
incorrect_indices = np.nonzero(np.argmax(y_pred,axis=1) != np.argmax(validation_data[1],axis=1))[0]

labels = ['dog', 'cat']

image = 7
plt.imshow(validation_data[0][incorrect_indices[image]].reshape(50,50), cmap=plt.get_cmap('gray'))
plt.show()
print("Prediction: {0}".format(labels[np.argmax(y_pred[incorrect_indices[image]])]))

image = 3
plt.imshow(validation_data[0][correct_indices[image]].reshape(50,50), cmap=plt.get_cmap('gray'))
plt.show()
print("Prediction: {0}".format(labels[np.argmax(y_pred[correct_indices[image]])]))
