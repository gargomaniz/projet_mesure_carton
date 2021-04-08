# -*- coding: utf-8 -*-

# IMPORTS
    
import os
import zipfile
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
from numpy import asarray

from PIL import Image


import tensorflow as tf
from matplotlib import pyplot as plt
#import tensorflow_datasets as tfds
import seaborn as sns

import time
import PIL

from process_masks import mask_processor, load_from_opencv_xml

print("Tensorflow version " + tf.__version__)


#Define classes
class_names = ['background', 'box'] 

# generate a list that contains one color for each class
colors = sns.color_palette(None, len(class_names))

"""Function to give a color to the segmentation masK"""

def give_color_to_annotation(annotation):
  '''
  Converts a 2-D annotation to a numpy array with shape (height, width, 3) where
  the third axis represents the color channel. The label values are multiplied by
  255 and placed in this axis to give color to the annotation

  Args:
    annotation (numpy array) - label map array
  
  Returns:
    the annotation array with an additional color channel/axis
  '''
  seg_img = np.zeros( (annotation.shape[0],annotation.shape[1], 3) ).astype('float')
  
  print(annotation.shape)
  for c in range(2):
    segc = (annotation == c)
    seg_img[:,:,0] += segc*( colors[c][0] * 255.0)
    seg_img[:,:,1] += segc*( colors[c][1] * 255.0)
    seg_img[:,:,2] += segc*( colors[c][2] * 255.0)
  
  return seg_img

"""### **Load the model**"""

reconstructed_model = tf.keras.models.load_model("./unet_model.h5") #Path to the model

"""### **Image to predict**"""

#load image to predict and resize to shape (128, 128, 3)
image_path1 = './carton_7_pos1.jpg' #Image that we wabt to predict segmentation mask
image_path2 = './carton_7_pos2.jpg'
"""### **Predict segmentation mask**"""

height=128
width=128

img_raw = tf.io.read_file(image_path1)
image = tf.image.decode_jpeg(img_raw)

image = tf.image.resize(image, (height, width))
image = tf.reshape(image, (height, width, 3,))
image = tf.cast(image, dtype=tf.float32) /255.0

#resize shape because UNet input is (None, 128,128,3)
images = tf.reshape(image, (1,height, width, 3,))


#prediction of the image and reformating output from (128,128,2) to (128,128,1)
prediction = reconstructed_model.predict(images)

prediction= np.argmax(prediction, axis=3)

#changing values of output from binary to colorized image for display
pred_img = give_color_to_annotation(prediction[0])

image = image*255
images1 = np.uint8([image, pred_img])


#passing image from camera 2 to the Network

img_raw = tf.io.read_file(image_path2)
image = tf.image.decode_jpeg(img_raw)

image = tf.image.resize(image, (height, width))
image = tf.reshape(image, (height, width, 3,))
image = tf.cast(image, dtype=tf.float32) /255.0

#resize shape because UNet input is (None, 128,128,3)
images = tf.reshape(image, (1,height, width, 3,))

#prediction of the image and reformating output from (128,128,2) to (128,128,1)
prediction = reconstructed_model.predict(images)

prediction= np.argmax(prediction, axis=3)

#changing values of output from binary to colorized image for display
pred_img = give_color_to_annotation(prediction[0])

image = image*255
images2 = np.uint8([image, pred_img])

images = images1 + images2
#displaying image with predicted mask
plt.figure(figsize=(10,4))

for idx, im in enumerate(images2):
  plt.subplot(2, 2, idx+1)
  plt.xticks([])
  plt.yticks([])
  #plt.title(titles[idx], fontsize=12)
  plt.imshow(im)
  plt.show()

H12 = load_from_opencv_xml("homography2to1.xml", "H")

Hv1_1 = load_from_opencv_xml("Hv1_1.xml", "Hv1_1")
Hv1_2 = load_from_opencv_xml("Hv1_2.xml", "Hv1_2")

lines, h, l = mask_processor(images1[0], images1[1], images2[0], images2[1], H12, Hv1_1, Hv1_2)

print("h", h)
print("l", l)
plt.imshow(lines)
plt.show()
print(type(im))

new_im = Image.fromarray(lines)

"""### **Save the segmentation mask**"""

new_im.save('./mask_carton.png') #Save segmentation mask
