import os
import random
import itertools
import numpy as np
import pandas as pd

import logging

import imgaug.augmenters as iaa

from matplotlib import gridspec
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import keras
from keras import layers, Input, models
from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def Data_Augmentation(wafer, label, iter):

    Augmented = []
    for i in range(len(wafer)): # i = Original Image
        for j in range(iter):   # k = Data Augmentation iteration per

            Filp_h = iaa.Sequential([iaa.Fliplr(1.0)])
            Filp_v = iaa.Sequential([iaa.Flipud(1.0)])
            Crop = iaa.CropAndPad(percent=(random.uniform(-0.25, -0.001), random.uniform(0.001, 0.25)))
            Roatate = iaa.Affine(rotate=(-45, 45))
            Shear = iaa.Affine(shear=(random.uniform(-30, -1), random.uniform(1, 30)))
            Translate = iaa.Affine(translate_percent={"x": (random.uniform(-0.1, -0.001), random.uniform(0.001, 0.1)), "y": (random.uniform(-0.1, -0.001), random.uniform(0.001, 0.1))})
            Blur = iaa.GaussianBlur(sigma=(0.03, 0.8))

            Aug = Filp_h.to_deterministic()
            image_aug = Aug.augment_images([wafer[i]])[0]
            Augmented.append(image_aug)

            Aug = Filp_v.to_deterministic()
            image_aug = Aug.augment_images([wafer[i]])[0]
            Augmented.append(image_aug)

            Aug = Crop.to_deterministic()
            image_aug = Aug.augment_images([wafer[i]])[0]
            Augmented.append(image_aug)

            Aug = Roatate.to_deterministic()
            image_aug = Aug.augment_images([wafer[i]])[0]
            Augmented.append(image_aug)

            Aug = Shear.to_deterministic()
            image_aug = Aug.augment_images([wafer[i]])[0]
            Augmented.append(image_aug)

            Aug = Translate.to_deterministic()
            image_aug = Aug.augment_images([wafer[i]])[0]
            Augmented.append(image_aug)

            Aug = Blur.to_deterministic()
            image_aug = Aug.augment_images([wafer[i]])[0]
            Augmented.append(image_aug)

    Label = np.full((len(Augmented), 1), label)

    return Augmented, Label


def Plot_Confusion_Matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def Create_Model():

    input_shape = (26, 26, 3)
    input_tensor = Input(input_shape)

    # 1st feature
    conv_1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_tensor)
    pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(conv_1)
    conv_2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pool_1)

    # 2nd feature
    conv_3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv_2)
    pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(conv_3)
    conv_4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool_2)

    # 3rd feature
    conv_5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv_4)
    pool_3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(conv_5)
    conv_6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool_3)

    # 4th feature
    conv_7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv_6)
    pool_4 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(conv_7)
    conv_8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool_4)

    drop_1 = layers.SpatialDropout2D(0.2)(conv_8)
    pool_5 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(drop_1)

    flat = layers.Flatten()(pool_5)

    dense_1 = layers.Dense(512, activation='relu')(flat)
    dense_2 = layers.Dense(128, activation='relu')(dense_1)
    output_tensor = layers.Dense(9, activation='softmax')(dense_2)

    model = models.Model(input_tensor, output_tensor)
    #Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model


def Main():
    # Set logger
    Logger = logging.getLogger()
    Logger.setLevel(logging.INFO)

    Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # 저장될 log의 format
    #Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Write Log at Console
    StreamHandler = logging.StreamHandler()
    StreamHandler.setFormatter(Formatter)
    Logger.addHandler(StreamHandler)

    # Save Log at my.log File
    FileHandler = logging.FileHandler('my.log')
    FileHandler.setFormatter(Formatter)
    Logger.addHandler(FileHandler)

    Logger.info(f'-------------------- Start Main --------------------')


    # Data Load
    pklPath = "Labeled_Data_1.pkl"
    Data = pd.read_pickle(pklPath)
    Logger.info(f'Data Loaded')


    # Data Pre-Process
    Data_26 = Data.loc[Data['waferMapDim'] == (26, 26)]
    sub_wafer = Data_26['waferMap'].values

    sw = np.ones((1, 26, 26))
    label = list()

    for i in range(len(Data_26)):
        # skip null label
        if len(Data_26.iloc[i, :]['failureType']) == 0:
            continue
        sw = np.concatenate((sw, Data_26.iloc[i, :]['waferMap'].reshape(1, 26, 26)))
        label.append(Data_26.iloc[i, :]['failureType'][0][0])

    x = sw[1:]                            # image, shape = (14366, 26, 26)
    y = np.array(label).reshape((-1, 1))  # label, shape = (14366, 1)      ex) y[2041] = ['none']

    x = x.reshape((-1, 26, 26, 1))
    FaultyCase = np.unique(y)       # ['Center' 'Donut' 'Edge-Loc' 'Edge-Ring' 'Loc' 'Near-full' 'Random' 'Scratch' 'none']

    new_x = np.zeros((len(x), 26, 26, 3))
    for w in range(len(x)):
        for i in range(26):
            for j in range(26):
                new_x[w, i, j, int(x[w, i, j])] = 1


    print("------------------ Before Augmented -----------------")
    print('new_X, Y shape : ({}, {})'.format(new_x.shape, y.shape))
    for F in FaultyCase:
        print('{} : {}'.format(F, len(y[y == F])))


    # Data Augmentation
    for F in FaultyCase:
        # skip none case
        if F == 'none':
            continue

        iter = int(1428 / len(new_x[np.where(y == F)[0]]) + 1)
        #iter = int(2000 / len(new_x[np.where(y == F)[0]]) + 1)

        gen_x, gen_y = Data_Augmentation(new_x[np.where(y == F)[0]], F, iter)
        new_x = np.concatenate((new_x, gen_x), axis=0)
        y = np.concatenate((y, gen_y))

    print("------------------ After Augmented ------------------")
    print('new_X, Y shape : ({}, {})'.format(new_x.shape, y.shape))
    for F in FaultyCase:
        print('{} : {}'.format(F, len(y[y == F])))

    Logger.info(f'Finish Data Augmentation')


    # Delete some None Class
    NoneIdx = np.where(y == 'none')[0][np.random.choice(len(np.where(y == 'none')[0]), size=3489, replace=False)]
    new_x = np.delete(new_x, NoneIdx, axis=0)
    new_y = np.delete(y, NoneIdx, axis=0)
    print('After Delete "none" class')
    print('new_X, Y shape : ({}, {})'.format(new_x.shape, new_y.shape))


    # Print All Class, Calculate Data Size
    DataSize = 0
    for F in FaultyCase:
        DataSize += len(new_y[new_y == F])
        print('{} : {}'.format(F, len(new_y[new_y == F])))

    for i, l in enumerate(FaultyCase):
        new_y[new_y == l] = i
    new_y = to_categorical(new_y)

    Train_Valid_X, Test_X, Train_Valid_X, Test_Y = train_test_split(new_x, new_y, test_size=0.15, random_state=2019, shuffle=True)
    Train_X, Valid_X, Train_Y, Valid_Y = train_test_split(new_x, new_y, test_size=0.2, random_state=2019, shuffle=True)
    Logger.info(f'Train x : {Train_X.shape}, Train y : {Train_Y.shape}')
    Logger.info(f'Valid x : {Valid_X.shape}, Valid y : {Valid_Y.shape}')
    Logger.info(f'Test  x : {Test_X.shape},  Test y  : {Test_Y.shape}')


    # Set Train Parameter
    Epoch = 30
    BatchSize = 1024


    # Train Model
    model = KerasClassifier(build_fn=Create_Model, epochs=Epoch, batch_size=BatchSize, verbose=2)

    # history
    Logger.info(f'Start Train')
    History = model.fit(Train_X, Train_Y,
                        validation_data=[Valid_X, Valid_Y],
                        epochs=Epoch,
                        batch_size=BatchSize,
                        )
    Logger.info(f'Finish Train')


    # Result - acc plot
    plt.plot(History.history['acc'])
    plt.plot(History.history['val_acc'])
    plt.title('model accuracy')
    plt.ylim([0.0, 1.0])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.png')
    plt.show()

    # Result - loss plot
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylim([0.0, 2.0])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()


    Test_Y_Pred = model.predict(Test_X)
    Test_Y = np.argmax(Test_Y, axis=1)  # one hot encoding to Class
    # https://stackoverflow.com/questions/54589669/confusion-matrix-error-classification-metrics-cant-handle-a-mix-of-multilabel
    print("Test_Y[:20]: ", Test_Y[:20])
    print("Test_Y_Pred[:20]: ", Test_Y_Pred[:20])

    cnf_matrix = confusion_matrix(Test_Y, Test_Y_Pred)
    np.set_printoptions(precision=2)
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # non normalized confusion matrix
    plt.subplot(gs[0])
    Plot_Confusion_Matrix(cnf_matrix, title='Confusion matrix')

    # normalized confusion matrix
    plt.subplot(gs[1])
    Plot_Confusion_Matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')

    plt.savefig("confusionMatrix.png")
    plt.show()

    Logger.info(f'Finish Main')


if __name__ == "__main__":
    Main()