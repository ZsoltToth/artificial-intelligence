'''
Testing the VGG16 Image Classifier

Based on the following tutorial:
https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/

'''
import keras
# pillow should be installed for load_img
#import pillow
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
model = VGG16()

image = load_img('./data/lion.jpg',target_size=(224,224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = preprocess_input(image)

predict = model.predict(image)

labels = decode_predictions(predict)

for label in labels[0]:
    print("%s (%.2f)" % (label[1], label[2] * 100.0))