#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import warnings
warnings.filterwarnings("ignore")

import sys
import getopt

try:
      options, argments= getopt.getopt(sys.argv[1:],"i:rh",["img_path=","random","help"])
except getopt.GetoptError:
      print ('nope ! try it that way if you dont mind \n plant-disease.py  -i|--img_path <path> | [-r|--random  ] | --help|-h ')
      sys.exit(2)


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot  as plt  
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Input
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from PIL import Image
import os
import numpy as np
import json


# In[ ]:
def print_prob(img,ps,classes,topk,title):
  fig,(ax1,ax2)=plt.subplots(figsize=(11,20),ncols=2)
  ax1.axis('off')
  img=(img*255).astype(np.uint8)
  ax1.imshow(img.squeeze())
  ax1.set_title(title)
  ax2.barh(np.arange(topk),ps,align='center',alpha=.5)
  ax2.set(yticks=range(topk),yticklabels=classes)
  ax2.set_aspect(0.1)
  ax2.set_xlim(0, 1.1)
  plt.tight_layout()
  plt.show()

# In[ ]:


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    prediction(model,img)
    


# In[ ]:


def random_image_from_testset():
        setting = dict(rescale=1./255,
                 #  preprocessing_function=preprocess_input,
                   horizontal_flip=True,
                   vertical_flip=True)
        train_datagen=ImageDataGenerator(**setting)

        test_generator=train_datagen.flow_from_directory(TEST_PATH, # this is where you specify the path to the main data folder
                                                         target_size=(224,224),
                                                         color_mode='rgb',
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         shuffle=True)

        images,labels=next(iter(test_generator))
        idx=labels[0].argsort().tolist()[-1:][::-1][0]
        class_=idx_to_class[str(idx)]
        print('ground truth {} '.format(class_))
        prediction(model,images[0],normalize=False,title=class_)


# In[ ]:


def prediction(model,img,topk=5,normalize=True,title="leaf"):
  x =np.expand_dims(img, axis=0) if img.shape[0]!=1 else img
  x= x*1./255 if normalize else x
 # x = preprocess_input(x)
  preds = model.predict(x)
  np.clip(x,0,1)  
  # decode the results into a list of tuples (class, description, probability)
  # (one such list for each sample in the batch)
  idx=preds.argsort().squeeze(0).tolist()[-topk:][::-1]
  top_pred=preds.squeeze(0)[idx]
  classes=[idx_to_class[str(x)] for x in idx]
  print_prob(x,top_pred,classes,topk,title=title)
  




def build_model(num_classes,input_shape=(224,224,3)):
  input_tensor = Input(shape=input_shape)
  base_model = VGG16(include_top=False,weights=None,input_tensor=input_tensor)
  x = base_model.output
  x = Flatten()(x)
  # let's add a fully-connected layer
  x = Dense(1024, activation='relu')(x)
  x = Dense(1024, activation='relu')(x)
  # and a logistic layer -- let's say we have 200 classes
  predictions = Dense(num_classes, activation='softmax')(x)
  # this is the model we will train
  model = Model(inputs=base_model.input, outputs=predictions)

  # i.e. freeze all convolutional InceptionV3 layers
  for layer in base_model.layers:
      layer.trainable = False
  return model    


# In[ ]:
    
with open('idx_to_class.json','r') as f:
    idx_to_class=json.load(f)
TEST_PATH='D:/downloads/plant-diseases/dataset_itr2/test'
model = build_model(38) 
model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('epochs_014-val_acc_0.954.hdf5')
for opt, arg in options:
    if opt in ('-h','--help'):
        print ('plant_detection.py -i <img_path> | -r  ')
        sys.exit()
    elif opt in ('-i','--img_path','-img_path'):
         predict_image(arg)
    elif opt in ('-r','--random'):
        random_image_from_testset()


# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




