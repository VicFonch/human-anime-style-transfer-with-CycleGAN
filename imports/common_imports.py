import numpy as np
import matplotlib.pyplot as plt
import os, shutil
from tqdm import tqdm
import cv2
import time
import lpips
from PIL import Image
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from itertools import islice

import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.layers import Input, BatchNormalization
from keras.layers import Concatenate, Add, Multiply, Average 
from keras.layers import Reshape, UpSampling2D, Conv2DTranspose, Lambda
from keras.layers import LeakyReLU, ReLU, PReLU, ELU, ThresholdedReLU, Softmax, Activation
from keras.layers import Layer
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

import tensorflow_datasets as tfds