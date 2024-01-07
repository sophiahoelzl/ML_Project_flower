import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pathlib
import PIL
import os
import os.path
import cv2
import random
import pandas as pd
import matplotlib.image as mpimg
import tensorflow as tf
from keras import Model
import keras
from matplotlib import pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing


from tensorflow.keras.optimizers.legacy import Adam
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout, \
    GlobalAveragePooling2D
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM, Bidirectional, Conv1D, concatenate, Permute, Reshape
from keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_labels(dir_path):
    label_dict = dict(zip(os.listdir(dir_path), range(len(dir_path))))
    print(dir_path + ": labels loaded...")
    return label_dict


def resize_images(dir_path):
    for (root, dirs, files) in os.walk(dir_path):
        for f in files:
            f_img = root + "/" + f
            img = Image.open(f_img)
            img = img.resize((256, 256))
            img.save(f_img)
    print(root + ": images resized...")


def load_images_from_files(dir_path, label_dict):
    print("load_images_from_files dirpath=" + dir_path + ": begin...")
    images = []
    labels = []

    for (root, dirs, files) in os.walk(dir_path):
        for f in files:
            # resizing images here to keep iterations smaller
            """
            f_img = root + "/" + f
            img = Image.open(f_img)
            img = img.resize((256,256))
            img.save(f_img)
            """

            file_path = os.path.join(root, f)
            img = mpimg.imread(file_path)
            normalized = np.asarray(img / 255)

            images.append(normalized)
            labels.append(label_dict[os.path.basename(os.path.dirname(os.path.join(root, f)))])
    print("load_images_from_files dirpath=" + dir_path + ": images loaded...")
    return images, labels


# def check_images():

"""
def display_random_img(target_dir, target_class):
    target_folder = target_dir + target_class
    random_image = random.sample(os.listdir(target_folder), 1)

    img = mpimg.imread(target_folder + "/" + random_image[0])

    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f"Image shape: {img.shape}")

    return img
"""


def random_forest(train_images, train_labels, test_images, test_labels):
    # reshape
    print("random_forest: reshaping...")
    train_nmbr = np.shape(train_images)[1] * np.shape(train_images)[2] * np.shape(train_images)[3]
    test_nmbr = np.shape(test_images)[1] * np.shape(test_images)[2] * np.shape(test_images)[3]

    train_imgs_flattened = np.reshape(train_images, (np.shape(train_images)[0], train_nmbr))
    test_imgs_flattened = np.reshape(test_images, (np.shape(test_images)[0], test_nmbr))

    train_imgs_flattened.shape, test_imgs_flattened.shape

    print("random_forest: constructing and training classifier...")
    rnf_clf = RandomForestClassifier(n_jobs=-1)
    rnf_clf.fit(train_imgs_flattened, train_labels, sample_weight=None)

    print("random_forest: making predictions using classifier...")
    predicted_labels = rnf_clf.predict(test_imgs_flattened)

    return predicted_labels


def cnn(train_dir, test_dir):
    print("cnn: creating generator...")
    imagegen = ImageDataGenerator()
    train_data = imagegen.flow_from_directory(train_dir, class_mode="categorical",
                                              batch_size=128, target_size=(256, 256))
    test_data = imagegen.flow_from_directory(test_dir, class_mode="categorical",
                                             batch_size=128, target_size=(256, 256), shuffle=False)

    model = Sequential()
    model.add(InputLayer(input_shape=(256, 256, 3)))
    # 1st conv block
    model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    # 2nd conv block
    model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    # 3rd conv block
    model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    # ANN block
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.25))
    # output layer
    model.add(Dense(units=5, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss = 'categorical_crossentropy',
        metrics=["accuracy", "Recall", "Precision"],
    )

    history = model.fit(train_data, epochs=1, validation_data=test_data)
    print(history.history.keys())

    model.summary()

    loss, accuracy, recall, precision = model.evaluate(test_data)
    print("Loss: %.2f %%" % (100*loss))
    print("Accuracy: %.2f %%" % (100*accuracy))
    print("Recall: %.2f %%" % (100*recall))
    print("Precision: %.2f %%"% (100*precision))

    return history

def print_metrics(test_labels, predicted_labels):
    print("Accuracy: %.2f %% " % (100.0 * accuracy_score(test_labels, predicted_labels)))
    print("Precision: %.2f %%" % (100.0 * precision_score(test_labels, predicted_labels, average='weighted')))
    print("Recall: %.2f %%" % (100.0 * recall_score(test_labels, predicted_labels, average='weighted')))
    print("F1: %.2f %%" % (100.0 * f1_score(test_labels, predicted_labels, average='weighted')))


def plot_loss_curves(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def confusion_matrix_graph(test_labels, predicted_labels, labels):
    cm = confusion_matrix(test_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels.keys())
    disp.plot(xticks_rotation='vertical')
    plt.show()

def lstm_pipe(in_layer):
    row_hidden = 128
    col_hidden = 128
    
    x = Conv1D(row_hidden, kernel_size=3, padding = 'same')(in_layer)
    x = Conv1D(row_hidden, kernel_size=3, padding = 'same')(x)
    
    x = Reshape((-1, row_hidden))(x)

    encoded_rows = Bidirectional(LSTM(row_hidden, return_sequences = True))(x)
    return LSTM(col_hidden)(encoded_rows)

def rnn(train_data, train_labels, test_data, test_labels):
    print("rnn: adding cnn layers...")

    batch_size = 32
    num_classes = 1
    epochs = 2

    row, col, channels = 256, 256, 3

    input = Input(shape=(row, col, channels))

    first_read = lstm_pipe(input)
    trans_read = lstm_pipe(Permute(dims = (2, 1, 3))(input))
    
    encoded_columns = concatenate([first_read, trans_read])
    encoded_columns = Dropout(0.2)(encoded_columns)
    
    prediction = Dense(num_classes, activation='softmax')(encoded_columns)
    
    model = Model(input, prediction)
    
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    model.summary()

    history = model.fit(train_data, train_labels,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(test_data, test_labels))
    
    loss, accuracy, recall, precision = model.evaluate(test_data)
    print("Loss: %.2f %%" % (100*loss))
    print("Accuracy: %.2f %%" % (100*accuracy))
    print("Recall: %.2f %%" % (100*recall))
    print("Precision: %.2f %%"% (100*precision))
    
    return history

def main():
    train_path = "archive/train"
    test_path = "archive/test"

    labels = load_labels(train_path)
    # test_labels = load_labels(test_path)

    # for resizing
    # resize_images(train_path)

    train_images, train_labels = load_images_from_files(train_path, labels)
    test_images, test_labels = load_images_from_files(test_path, labels)

    # img = display_random_img("archive/train/", random.choice(list(train_classes)))

    np.shape(train_images), np.shape(train_labels)

    # random forest
    #predictions = random_forest(train_images, train_labels, test_images, test_labels)
    #print_metrics(test_labels, predictions)
    #confusion_matrix_graph(test_labels, predictions, labels)

    # cnn
    #history = cnn(train_path, test_path)
    #plot_loss_curves(history)

    #pretrained_model = pre_trained_model()
    #pre_trained_history = train_pretrained_model(pretrained_model, train_path, test_path)
    #plot_loss_curves(pre_trained_history)

    # rnn
    rnn(np.array(train_images),
        np.array(train_labels),
        np.array(test_images),
        np.array(test_labels))

    return 0


## Pre-trained model ##

num_classes = 5


def pre_trained_model():
    model = get_model()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", "Recall", "Precision"],
    )

    model.summary()
    return model


def get_model():
    base_model = ResNet50(weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    base_model_ouput = base_model.output

    x = GlobalAveragePooling2D()(base_model_ouput)
    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax', name='fcnew')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def train_pretrained_model(model, train_dir, test_dir):
    image_size = 256
    batch_size = 128

    train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    valid_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_data_gen.flow_from_directory(train_dir, (image_size, image_size), batch_size=batch_size,
                                                         class_mode='categorical')
    valid_generator = valid_data_gen.flow_from_directory(test_dir, (image_size, image_size), batch_size=batch_size,
                                                         class_mode='categorical')

    # Training the model
    pre_trained_history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        validation_data=valid_generator,
        validation_steps=valid_generator.n // batch_size,
        epochs=10,
        verbose=1)

    print(pre_trained_history.history.keys())

    imagegen = ImageDataGenerator()
    test_data = imagegen.flow_from_directory(test_dir, class_mode="categorical", batch_size=128, target_size=(256, 256), shuffle=False)
    loss, accuracy, recall, precision = model.evaluate(test_data)
    print("Loss: %.2f %%" % (100 * loss))
    print("Accuracy: %.2f %%" % (100 * accuracy))
    print("Recall: %.2f %%" % (100 * recall))
    print("Precision: %.2f %%" % (100 * precision))

    return pre_trained_history


if __name__ == '__main__':
    main()
