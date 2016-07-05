from __future__ import print_function
import numpy as np
import os

from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
import cv2


class MNISTClassifier(object):

    def __init__(self, batch_size=128, nb_classes=10):
        self.img_rows, self.img_cols = 28, 28
        self.batch_size = batch_size
        self.nb_classes = nb_classes

    def _loadAndPreprocessTraining(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.img_rows, self.img_cols).astype('float32') / 255
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.img_rows, self.img_cols).astype('float32') / 255

        self.Y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
        self.Y_test = np_utils.to_categorical(self.y_test, self.nb_classes)

    def buildNN(self):
        if os.path.isfile("MNISTArchitecture.json"):
            print("  loading architecture")
            self.model = model_from_json(open('MNISTArchitecture.json').read())
        else:
            self.model = Sequential()

            self.model.add(Convolution2D(64, 5, 5, input_shape=(1, self.img_rows, self.img_cols)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Flatten())
            self.model.add(Dense(256))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.nb_classes))
            self.model.add(Activation('softmax'))

            mnistArchitecture = self.model.to_json()
            open('MNISTArchitecture.json', 'w').write(mnistArchitecture)
            pass

        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0000001, momentum=0.8, nesterov=True), metrics=['accuracy'])

        if os.path.isfile("MNISTWeights.h5"):
            print("  loading weights")
            self.model.load_weights('MNISTWeights.h5')

    def train(self, nb_epoch=24):
        self._loadAndPreprocessTraining()
        self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(self.X_test, self.Y_test))
        self.model.save_weights('MNISTWeights.h5', overwrite=True)

        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def predict(self, data):
        data = data.reshape(data.shape[0], 1, 28, 28).astype('float32') / 255
        return self.model.predict_classes(data, batch_size=32)

class Equation(object):
    def __init__(self, image):
        image = image[None, :, :]
        self.chars = []

        isImg = False
        height = len(image)
        chars = []
        firstX = 0
        for col in range(len(image.T)):
            if isImg:
                if np.count_nonzero(image.T[col]) == 0:
                    isImg = False
                    self.chars.append(image[0:28, firstX:col])
            else:
                if np.count_nonzero(image.T[col]) > 0:
                    isImg = True
                    firstX = col

    def getSplitEquation(self):
        return self.chars

classifier = MNISTClassifier()
print("Building Network")
classifier.buildNN()
classifier.train()
print("Using Network")

equ = Equation(cv2.imread('equation.png', 0))
for ch in equ.getSplitEquation():
    print(classifier.predict(ch))
# print(equ.getSplitEquation())
#
#
# isImg = False
# height = len(img)
#
# chars = []
# firstX = 0
#
# for col in range(len(img.T)):
#     if isImg:
#         if np.count_nonzero(img.T[col]) == 0:
#             isImg = False
#             print((img[0:28, firstX:col]).shape)
#             chars.append(img[0:28, firstX:col])
#     else:
#         if np.count_nonzero(img.T[col]) > 0:
#             isImg = True
#             firstX = col
#
# a = np.append(chars[1].T, np.zeros((16, 28))).reshape(28, 28).T
# print(classifier.predict(char[0])[a, :, :])
# # print("LOOK HERE", a.shape)
# # print(classifier.predict(a[None, :, :]))
# plt.imshow(chars[0], cmap="Greys_r", interpolation="none")
# plt.show()
