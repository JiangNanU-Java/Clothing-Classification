# Classfication of Fashion MNIST data into 1 of the 10
# categories using a neural network.

# Importing tensorflow
import cv2

import tensorflow as tf
from PIL import Image
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


def produceImage(file_in, width, height, file_out):
    """
    像素格式缩放
    :param file_in:
    :param width:
    :param height:
    :param file_out:
    :return:
    """
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)


def grayImg(imgName):
    """
    灰度图
    :param imgName:
    :return:
    """
    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(imgName, img)


def img2darray(filepath):
    """
    以ndarray格式读取图片
    :param filepath:
    :return:
    """
    img = Image.open(filepath)
    return np.array(img)


imgPath = '4.png'

# 缩放
file_in = imgPath
width = 28
height = 28
file_out = imgPath
produceImage(file_in, width, height, file_out)

# 灰度图
grayImg(imgPath)

# Loading the fashion mnist data

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images2, test_labels2) = fashion_mnist.load_data()

print(img2darray(imgPath))
test_images = np.array([img2darray(imgPath)])
test_labels = np.array([1])


def display_image(plt, img):
    plt.imshow(img, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()

# Class name corresponding to the labels.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Checking the dimensions of train and test sets.

print("Train set dimensions:")
print(train_images.shape)

print("Test set dimensions")
print(test_images.shape)

# Displaying the first image in the train set.
plt.figure()

display_image(plt, train_images[0])

# Normalizing the images in the test and train set.
train_images = train_images / 255.0
test_images = test_images / 255.0

display_image(plt, train_images[0])

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()

# Defining an artificial neural network model
# The first layers takes input as a flatten array of the 2d image.
# The second layer is used as hidden layer with 128 neuron.
# Each neuron has a relu activation function.
# Each neuron is connected to each input neuron of the previous layer hence Dense layer
# The third layer is the output layer which uses the input from the previous layer and
# uses the softmax activation function to squeeze the input into set of 10 probabilities values
# Each for the classes mentioned above.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # This layer is the input layer
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(
        "{} {:2.0f}% {}".format(class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]))


def plot_value_array(i, predictions_array, true_label, class_names):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    prediction_label = np.argmax(predictions_array)

    this_plot[prediction_label].set_color('red')
    this_plot[true_label].set_color('blue')


#
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions, test_labels, test_images, class_names)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions, test_labels, class_names)

num_rows = 1
num_cols = 1
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images, class_names)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels, class_names)

plt.show()
