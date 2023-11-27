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


from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout



def load_labels(dir_path):
    label_dict =  dict(zip(os.listdir(dir_path), range(len(dir_path))))
    print(dir_path + ": labels loaded...")
    return label_dict


def resize_images(dir_path):
    for (root,dirs,files) in os.walk(dir_path):
        for f in files:
            f_img = root + "/" + f
            img = Image.open(f_img)
            img = img.resize((256,256))
            img.save(f_img)
    print(root + ": images resized...")

def load_images_from_files(dir_path, label_dict):
    print("load_images_from_files dirpath=" + dir_path + ": begin...") 
    images = []
    labels = []

    for (root,dirs,files) in os.walk(dir_path):
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
            normalized = np.asarray(img/255)
            
            images.append(normalized)
            labels.append(label_dict[os.path.basename(os.path.dirname(os.path.join(root,f)))])
    print("load_images_from_files dirpath=" + dir_path + ": images loaded...") 
    return images, labels

#def check_images():

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
    #reshape
    print("random_forest: reshaping...")
    train_nmbr = np.shape(train_images)[1] * np.shape(train_images)[2] * np.shape(train_images)[3]
    test_nmbr = np.shape(test_images)[1] * np.shape(test_images)[2] * np.shape(test_images)[3]

    train_imgs_flattened = np.reshape(train_images, (np.shape(train_images)[0], train_nmbr))
    test_imgs_flattened = np.reshape(test_images, (np.shape(test_images)[0], test_nmbr))

    train_imgs_flattened.shape, test_imgs_flattened.shape

    print("random_forest: constructing and training classifier...")
    rnf_clf =  RandomForestClassifier(n_jobs=-1)
    rnf_clf.fit(train_imgs_flattened, train_labels, sample_weight=None)

    print("random_forest: making predictions using classifier...")
    predicted_labels =  rnf_clf.predict(test_imgs_flattened)

    return predicted_labels

def cnn(train_path):
    print("cnn: creating generator...")
    imagegen = ImageDataGenerator()
    print("cnn: loading training data...")
    train = imagegen.flow_from_directory("archive/train/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(256, 256))
    print("cnn: loading testing data...")
    test = imagegen.flow_from_directory("archive/test/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(256, 256))

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

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # fit on data for 30 epochs
    model.fit(train, epochs=30, validation_data=test)

    model.summary()

def print_metrics(test_labels, predicted_labels):
    print("Accuracy: %.2f %% " % (100.0*accuracy_score(test_labels, predicted_labels)))
    print("Precision: %.2f %%" % (100.0*precision_score(test_labels, predicted_labels, average='weighted')))
    print("Recall: %.2f %%" % (100.0*recall_score(test_labels, predicted_labels, average='weighted')))
    print("F1: %.2f %%" % (100.0*f1_score(test_labels, predicted_labels, average='weighted')))

def confusion_matrix(test_labels, predicted_labels, labels):
    cm = confusion_matrix(test_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_dict_train.keys())
    disp.plot(xticks_rotation='vertical')
    plt.show()

#loss function graphisch
#file-upload
#pre-trained model

def main():
    train_path = "archive/train"
    test_path = "archive/test"

    labels = load_labels(train_path)
    #test_labels = load_labels(test_path)

    #resize_images(train_path)

    train_images, train_labels = load_images_from_files(train_path, labels)
    test_images, test_labels = load_images_from_files(test_path, labels)

    #img = display_random_img("archive/train/", random.choice(list(train_classes)))

    np.shape(train_images), np.shape(train_labels)
    
    predictions = random_forest(train_images, train_labels, test_images, test_labels)

    print_metrics(test_labels, predictions, labels)
    return 0

if __name__ == '__main__':
    main()