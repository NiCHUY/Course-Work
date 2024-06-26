# -*- coding: utf-8 -*-
"""satellite-images-semantic-segmentation-by-u-net.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VeoqUt37nSWBLMKletx44AicatROFe1C

# Satellite Images Semantic Segmentation by U-Net
This noteboook referred to the following notebook.<br/>
https://www.kaggle.com/code/dikshabhati2002/image-segmentation-u-net

### About U-Net
from https://arxiv.org/abs/1505.04597<br/>
The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks.

# Import libraries
"""
import display as display
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2

"""# Dataset Preparation"""

dir0 = '../input/satellite-images-of-water-bodies/Water Bodies Dataset/Images'
dir1 = '../input/satellite-images-of-water-bodies/Water Bodies Dataset/Masks'

files = []
paths = []
for dirname, _, filenames in os.walk(dir0):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        paths.append(path)
        files.append(filename)

mfiles = []
mpaths = []
for dirname, _, filenames in os.walk(dir1):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        mpaths.append(path)
        mfiles.append(filename)

import cv2

path0 = 'D:/Water Bodies Dataset/Images/water_body_1.jpg'
img0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
shape0 = img0.shape
print(shape0)
plt.imshow(img0)
plt.show()

path1 = 'D:/Water Bodies Dataset/Masks/water_body_1.jpg'
img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
shape1 = img1.shape
print(shape1)
plt.imshow(img1)
plt.show()

"""dfi=df0[df0['new_filename']=='1000_train_1_.png'].reset_index(drop=True)
display(dfi)
for i in range(len(dfi)):
    path=dfi.loc[i,'mpath']
    if type(path)!=float:
        img=cv2.imread(dfi.loc[i,'mpath'],cv2.IMREAD_GRAYSCALE)
        print(img.shape)
        plt.title(dfi.loc[i,'label'])
        plt.imshow(img)
        plt.show()
"""

df0 = pd.DataFrame(columns=['file', 'path', 'mpath', 'class'])
df0['file'] = files
df0['path'] = paths
df0['mpath'] = mpaths
df0['class'] = 1

display(df0)
df0.sample(frac=1)

"""# train/test data setting"""

n = len(df0)
df = df0.iloc[0:(n // 10) * 3]
test_df = df0.iloc[(n // 10) * 3:(n // 10) * 4]

img_size = [256, 256]


def data_augmentation(car_img, mask_img):
    if tf.random.uniform(()) > 0.5:
        car_img = tf.image.flip_left_right(car_img)
        mask_img = tf.image.flip_left_right(mask_img)

    return car_img, mask_img


def preprocessing(car_path, mask_path):
    car_img = tf.io.read_file(car_path)
    car_img = tf.image.decode_jpeg(car_img, channels=3)
    car_img = tf.image.resize(car_img, img_size)
    car_img = tf.cast(car_img, tf.float32) / 255.0

    mask_img = tf.io.read_file(mask_path)
    mask_img = tf.image.decode_jpeg(mask_img, channels=3)
    mask_img = tf.image.resize(mask_img, img_size)
    mask_img = mask_img[:, :, :1]
    mask_img = tf.math.sign(mask_img)

    return car_img, mask_img


def create_dataset(df, train=False):
    if not train:
        ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["mpath"].values))
        ds = ds.map(preprocessing, tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["mpath"].values))
        ds = ds.map(preprocessing, tf.data.AUTOTUNE)
        ds = ds.map(data_augmentation, tf.data.AUTOTUNE)

    return ds


# error
print(df)

train_df, valid_df = train_test_split(df, random_state=42, test_size=.25)
train = create_dataset(train_df, train=True)
valid = create_dataset(valid_df)
test = create_dataset(test_df)

TRAIN_LENGTH = len(train_df)
BATCH_SIZE = 16
BUFFER_SIZE = 1000

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
valid_dataset = valid.batch(BATCH_SIZE)
test_dataset = test.batch(BATCH_SIZE)


def display(display_list):
    plt.figure(figsize=(12, 12))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


for i in range(5):
    for image, mask in train.take(i):
        sample_image, sample_mask = image, mask
        display([sample_image, sample_mask])

"""# MobileNetV2 Model"""

base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())

    return result


up_stack = [
    upsample(512, 3),  # 4x4 -> 8x8
    upsample(256, 3),  # 8x8 -> 16x16
    upsample(128, 3),  # 16x16 -> 32x32
    upsample(64, 3),  # 32x32 -> 64x64
]


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, activation='sigmoid',
        padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


"""# Train the Model"""


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_loss(in_gt, in_pred):
    return 1 - dice_coef(in_gt, in_pred)


model = unet_model(1)

model.compile(optimizer='adam',
              loss=dice_loss,
              metrics=[dice_coef, 'binary_accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)


def visualize(display_list):
    plt.figure(figsize=(12, 12))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions(sample_image, sample_mask):
    pred_mask = model.predict(sample_image[tf.newaxis, ...])
    pred_mask = pred_mask.reshape(img_size[0], img_size[1], 1)
    visualize([sample_image, sample_mask, pred_mask])


for i in range(5):
    for images, masks in train_dataset.take(i):
        for img, mask in zip(images, masks):
            sample_image = img
            sample_mask = mask
            show_predictions(sample_image, sample_mask)
            break

# model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + 1) % 3 == 0:
            show_predictions(sample_image, sample_mask)


EPOCHS = 40
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=valid_dataset,
                          callbacks=[DisplayCallback(), early_stop])

for i in range(5):
    for images, masks in test_dataset.take(i):
        for img, mask in zip(images, masks):
            tsample_image = img
            tsample_mask = mask
            show_predictions(tsample_image, tsample_mask)
            break
