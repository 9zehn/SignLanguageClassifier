import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras.utils  import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

"""
# Using keras sequential model, created a CNN to classify sign language
# from the MNIST databank.
# @Author: Leon Gruber
# @Date: October 2023
"""

class SignLanguage:
    def __init__(self):
        self.model = None

        self.data = {
            "train": None,
            "test" : None
        }
        self.create_model()

    def create_model(self):
        # Creating the model

        model = keras.Sequential(
            [
             keras.Input(shape=(28,28,1)),
             Conv2D(32,3,activation="relu"),
             Conv2D(32,3,activation="relu"),
             MaxPooling2D(3),
             Conv2D(32,3,activation="relu"),
             MaxPooling2D(2),
             Flatten(),
             Dropout(0.5),
             Dense(25,activation="softmax")
            ]
        )

        model.compile('adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

        self.model = model


    def prepare_data(self, images, labels):
        """
        :param images numpy array of size (num_examples, 28*28)
        :param labels numpy array of size (num_examples, )
        """

        # Split training and validation set
        # reshaping each example into a 2D image (28, 28, 1)

        X_train, X_test, y_train, y_test = train_test_split(images,labels,test_size=0.1)

        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255

        X_train = X_train.reshape(X_train.shape[0],28,28,1)
        X_test = X_test.reshape(X_test.shape[0],28,28,1)

        y_train = to_categorical(y_train, 25)
        y_test = to_categorical(y_test, 25)

        self.data = {
            "train": (X_train, y_train),
            "test" : (X_test, y_test)
        }


    def train(self, batch_size:int=128, epochs:int=50, verbose:int=1):
        """
        :param batch_size The batch size to use for training
        :param epochs     Number of epochs to use for training
        :param verbose    Whether or not to print training output
        """
        test_data = (self.data["test"][0],self.data["test"][1])

        history = self.model.fit(self.data["train"][0], self.data["train"][1], batch_size=batch_size, epochs=epochs, validation_data=test_data)

        if verbose:
          print(history)

        return history


    def predict(self, data):
        """
        :param data: numpy array of test images
        :return a numpy array of test labels. array size = (num_examples, )
        """

        # Normalizing data and predicting

        data = data.astype("float32") / 255

        data_formated = data.reshape(data.shape[0],28,28,1)


        predictions = self.model.predict(data_formated)
        pred = np.argmax(predictions,axis=1)


        return pred#np.zeros(data.shape[0])


    def visualize_data(self, data):
        """
        :param data: numpy array of images
        """

        if data is None: return

        nrows, ncols = 5, 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)

        for i in range(nrows):
            for j in range(ncols):
                axs[i][j].imshow(data[0][i*ncols+j].reshape(28, 28), cmap='gray')
        plt.show()


    def visualize_accuracy(self, history):
        """
        :param history: return value from model.fit()
        """
        if history is None: return

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title("Accuracy")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train','test'])
        plt.show()


if __name__=="__main__":
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')

    train_labels, test_labels = train['label'].values, test['label'].values
    train.drop('label', axis=1, inplace=True)
    test.drop('label', axis=1, inplace=True)

    num_classes = test_labels.max() + 1
    train_images, test_images = train.values, test.values

    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

    my_model = SignLanguage()
    my_model.prepare_data(train_images, train_labels)

    my_model.visualize_data(my_model.data["train"])

    history = my_model.train(epochs=30, verbose=1)
    my_model.visualize_accuracy(history)

    y_pred = my_model.predict(test_images)
    accuracy = accuracy_score(test_labels, y_pred)
    print(accuracy)
