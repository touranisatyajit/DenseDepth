import os
import glob
import argparse
import matplotlib
from PIL import Image

import cv2
# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt
import numpy as np
import time
# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='/home/tourani/Desktop/datasets/oxford/rear_undistort/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

def getDepth():

    outputs = predict(model, inputs)
    print(inputs.shape, outputs.shape)
    outputs = outputs.squeeze()
    outputs = cv2.resize(outputs,(1280,960),interpolation=cv2.INTER_CUBIC)
    outputs_np = np.array(outputs)
    outputs_np = np.abs(outputs)
    


        

foldername ='/home/tourani/Desktop/2014-06-26-09-24-58/front_undistort/'
folder_depth = '/home/tourani/Desktop/2014-06-26-09-24-58/front_undistort_depth/'
for filename in os.listdir(foldername):
        ff = str(foldername) + filename
        x = np.clip(np.asarray(Image.open( ff ), dtype=float) / 255, 0, 1)
        outputs = predict(model, x)
        outputs = outputs.squeeze()
        outputs = cv2.resize(outputs,(1280,960),interpolation=cv2.INTER_CUBIC)
        outputs_np = np.array(outputs)
        outputs_np = np.abs(outputs)
        file_name = folder_depth + filename.split('.')[0] + '.npy'
        #print(file_name)
        #time.sleep(40)
        np.save(file_name, outputs_np)
        


