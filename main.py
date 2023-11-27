import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pathlib
import PIL
import os
import os.path
import cv2
import random
import matplotlib.image as mpimg
import tensorflow as tf
from keras import Model
from tensorflow.keras.layers.experimental import preprocessing


from tensorflow.keras.optimizers.legacy import Adam
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout, \
    GlobalAveragePooling2D
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
    print("cnn: loading training data...")
    train = imagegen.flow_from_directory("archive/train/", class_mode="categorical", shuffle=False, batch_size=128,
                                         target_size=(256, 256))
    print("cnn: loading testing data...")
    test = imagegen.flow_from_directory("archive/test/", class_mode="categorical", shuffle=False, batch_size=128,
                                        target_size=(256, 256))

    BATCH_SIZE = 128
    train_data = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, image_size=(256, 256),
                                                             labels='inferred', label_mode='categorical',
                                                             batch_size=BATCH_SIZE)
    test_data = tf.keras.utils.image_dataset_from_directory(test_dir, shuffle=False, image_size=(256, 256),
                                                            labels='inferred', label_mode='categorical',
                                                            batch_size=BATCH_SIZE)

    '''data_augmentation = tf.keras.Sequential([
        # we use the model to rescale the data
        tf.keras.layers.Rescaling(scale=1. / 255, input_shape=(256, 256, 3)),
        # we tried to add more preporcessing layers, however they slowed down the whole
        # learning process to much which is why we didn't use them in the end
        # tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        # tf.keras.layers.RandomRotation(0.5),
    ])
    inputs = tf.keras.layers.Input(shape=(256, 256, 3), name='input_layer')
    sequential = data_augmentation(inputs)

    conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(sequential)
    maxPool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_1)
    # then 64
    conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(maxPool_1)
    maxPool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_2)
    # then 128
    conv_3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")(maxPool_2)
    # then we flattend the all the layers
    flat = tf.keras.layers.Flatten()(conv_3)
    # and use some dropout (40%) to prevent overfitting
    dropout_1 = tf.keras.layers.Dropout(0.4)(flat)
    # then we add a hidden layer
    dense_1 = tf.keras.layers.Dense(128, activation="relu")(dropout_1)
    # another for the output layer
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(dense_1)

    # then we build the model
    model = tf.keras.Model(inputs, outputs, name="training_model")'''
    model = Sequential()

    # defines the convolutional base
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
    # to perform classification
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=5, activation='softmax'))

    # model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'recall', 'precision'])

    print('hier')
    history = model.fit(train_data, validation_data=test_data, epochs=2)
    #history = model.fit(train_data, test_data, epochs=2)

    # for evaluation

    return history


def print_metrics(test_labels, predicted_labels):
    print("Accuracy: %.2f %% " % (100.0 * accuracy_score(test_labels, predicted_labels)))
    print("Precision: %.2f %%" % (100.0 * precision_score(test_labels, predicted_labels, average='weighted')))
    print("Recall: %.2f %%" % (100.0 * recall_score(test_labels, predicted_labels, average='weighted')))
    print("F1: %.2f %%" % (100.0 * f1_score(test_labels, predicted_labels, average='weighted')))


def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    recall = history.history['recall']
    val_recall = history.history['val_recall']

    precision = history.history['precision']
    val_precision = history.history['val_precision']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot recall
    plt.figure()
    plt.plot(epochs, recall, label='training_recall')
    plt.plot(epochs, val_recall, label='val_recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot precision
    plt.figure()
    plt.plot(epochs, precision, label='training_precision')
    plt.plot(epochs, val_precision, label='val_precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.legend()


def confusion_matrix_graph(test_labels, predicted_labels, labels):
    cm = confusion_matrix(test_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels.keys())
    disp.plot(xticks_rotation='vertical')
    plt.show()


# loss function graphisch
# file-upload
# pre-trained model

def main():
    train_path = "archive/train"
    test_path = "archive/test"

    # labels = load_labels(train_path)
    # test_labels = load_labels(test_path)

    # for resizing
    # resize_images(train_path)

    # train_images, train_labels = load_images_from_files(train_path, labels)
    # test_images, test_labels = load_images_from_files(test_path, labels)

    # img = display_random_img("archive/train/", random.choice(list(train_classes)))

    # np.shape(train_images), np.shape(train_labels)

    # random forest
    '''predictions = random_forest(train_images, train_labels, test_images, test_labels)
    print_metrics(test_labels, predictions)
    confusion_matrix_graph(test_labels, predictions, labels)'''

    # cnn
    history = cnn(train_path, test_path)
    plot_loss_curves(history)

    pretrained_model = pre_trained_model()
    train_pretrained_model(pretrained_model, train_path, test_path)
    return 0


## Pre-trained model ##

num_classes = 5


def pre_trained_model():
    model = get_model()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
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

    epochs = 5

    # Training the model

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        validation_data=valid_generator,
        validation_steps=valid_generator.n // batch_size,
        epochs=epochs,
        verbose=1)


if __name__ == '__main__':
    main()
