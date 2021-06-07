import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as implt
from PIL import Image 
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

train_dataset='train'
#test_dataset='test'

train_covid='train/Covid'
train_normal='train/Normal'

# plot some samples
covid1=implt.imread(train_covid+'/Covid (1).png')
plt.subplot(1,2,1)
plt.title('Covid Sample')
plt.imshow(covid1)
plt.show()
print(covid1.shape)


img_size=64
covid_train=[]
normal_train=[]
#viral_pneumonia_train=[]

#preprocessin
#train
covid_labels=np.ones(1252)
normal_labels=np.zeros(1229)

for i in os.listdir(train_covid):
     if os.path.isfile(train_dataset + "/Covid/" + i):
            covid=Image.open(train_dataset+'/Covid/'+i).convert('L') # converting grey scale
            covid=covid.resize((img_size,img_size),Image.ANTIALIAS) # resizing to 50,50
            covid=np.asarray(covid)/255 # bit format
            covid_train.append(covid)
            
for i in os.listdir(train_normal):
     if os.path.isfile(train_dataset + "/Normal/" + i):
            normal=Image.open(train_dataset+'/Normal/'+i).convert('L') # converting grey scale
            normal=normal.resize((img_size,img_size),Image.ANTIALIAS) # resizing to 50,50
            normal=np.asarray(normal)/255 # bit format
            normal_train.append(normal)

X_train=np.concatenate((covid_train,normal_train),axis=0)
Y_train=np.concatenate((covid_labels,normal_labels),axis=0).reshape(X_train.shape[0],1)
print('X train shape:',X_train.shape)
print('Y train shape:',Y_train.shape)

X_train = X_train.reshape(-1,img_size,img_size,1)
print("x_train shape: ",X_train.shape)


# Label Encoding
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 2)

# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_one,X_test,Y_one,Y_test=train_test_split(X_train,Y_train,test_size=0.2,random_state=2)
X_train,X_val,Y_train,Y_val=train_test_split(X_one,Y_one,test_size=0.2,random_state=2)


print("x_train shape",X_train.shape)
print("x_test shape",X_test.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_test.shape)
print("y_val shape",Y_val.shape)
print("y_val shape",Y_val.shape)

y_test_oned=[]

for i in range(497):
  if Y_test[i,0]==1:
      y_test_oned.append(0)
  else:
      y_test_oned.append(1)

y_test_oned = np.array(y_test_oned)

#model creation
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (img_size,img_size,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))
#padding valid dene
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation = "softmax"))


from keras.utils.vis_utils import plot_model
plot_model(model, to_file="model.png", show_shapes= True)


# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# Compile the model
model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])

epochs = 300  
batch_size = 25


# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

#creating early stopping function
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

#start the process
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)
                              

plt.figure(figsize = (10,5))
plt.plot(history.history["val_loss"], color="b", label = "validation loss")
plt.title("loss")
plt.legend()
plt.show()
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['val_loss', 'loss'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.save('bitirme4.model')

test_sonuclari = model.predict(X_test)

test_one=[]

for i in range(497):
  if test_sonuclari[i,0]>test_sonuclari[i,1]:
      test_one.append(0)
  else:
      test_one.append(1)

test_one = np.array(test_one)


cm_test = confusion_matrix(y_test_oned, test_one)

print(cm_test)




















