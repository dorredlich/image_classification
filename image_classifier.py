import numpy as np

import glob
import tensorflow as tf
import cv2

# global variables
TOTAL_DOGS_TRAIN = 1199
TOTAL_CATS_TRAIN = 1149
TOTAL_PIXELS_PIC = 65536
TOTAL_DOGS_TEST = 159
TOTAL_CATS_TEST = 149



def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))


def making_data_x_from_train():
    train_dogs = glob.glob("dogs vs cats/training_set/dogs/*.jpg")
    train_cats = glob.glob("dogs vs cats/training_set/cats/*.jpg")
    TOTAL_DOGS_TRAIN = len(train_dogs)
    TOTAL_CATS_TRAIN = len(train_cats)
    data_x1 = np.array([[np.array(cv2.cvtColor(cv2.imread(dogs), cv2.COLOR_RGB2GRAY))] for dogs in train_dogs])
    data_x2 = np.array([[np.array(cv2.cvtColor(cv2.imread(cats), cv2.COLOR_RGB2GRAY))] for cats in train_cats])
    data_x_tr = np.concatenate([data_x1, data_x2]) # מאחד את מערך של הפיקסלים של הכלבים עם מערך של פקסלים של החתולים
    data_x_tr = np.array([np.array(mat1.ravel()) for mat1 in data_x_tr]) # הופך אותו לפחות שני מימדים
    data_x_tr = np.array([a / 255. for a in data_x_tr]) # הופך כל פיקסל שנמצא בין 0 ל 255 לפקסלים בין 0 ל 1
    return data_x_tr


def making_data_y():
    dogs = np.array([[1] for x in range(TOTAL_DOGS_TRAIN)])  # number of pics in train dogs
    cats = np.array([[0] for y in range(TOTAL_CATS_TRAIN)])  # number of pics in train cats
    data_y_in = np.concatenate([dogs, cats])
    return data_y_in


features = TOTAL_PIXELS_PIC  # number of pixels in each pic
eps = 1e-12
x = tf.compat.v1.placeholder(tf.float32, [None, features])
y_ = tf.compat.v1.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([features, 1]))
b = tf.Variable(tf.zeros([1]))
y = 1 / (1.0 + tf.exp(-(tf.matmul(x, W) + b)))
loss1 = -(y_ * tf.math.log(y + eps) + (1 - y_) * tf.math.log(1 - y + eps))
loss = tf.reduce_mean(loss1)
update = tf.compat.v1.train.GradientDescentOptimizer(0.0001).minimize(loss)

data_x = making_data_x_from_train()
label = making_data_y()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(0, 10):
    sess.run(update, feed_dict={x: data_x, y_: label})  # BGD



test_dogs = glob.glob("dogs vs cats/test_set/dogs/*.jpg")
test_cats = glob.glob("dogs vs cats/test_set/cats/*.jpg")
data_test_dogs = np.array([[np.array(cv2.cvtColor(cv2.imread(dog), cv2.COLOR_RGB2GRAY))] for dog in test_dogs])
data_test_cats = np.array([[np.array(cv2.cvtColor(cv2.imread(cat), cv2.COLOR_RGB2GRAY))] for cat in test_cats])
print("first: ", data_test_dogs.shape)

data_test_dogs_flat = np.array([np.array(mat.ravel()) for mat in data_test_dogs])
data_test_cats_flat = np.array([np.array(mat.ravel()) for mat in data_test_cats])
data_test_dogs_flat = np.array([a / 255. for a in data_test_dogs_flat])
data_test_cats_flat = np.array([a / 255. for a in data_test_cats_flat])


sum = 0
for cat in data_test_cats_flat:
    sum += logistic_fun(np.matmul(np.array([cat]), sess.run(W)) + sess.run(b))[0][0]
print("sum: ", sum)
print("Prediction that it is cat on cats pictures: ", sum / TOTAL_CATS_TEST)

sum = 0
for dog in data_test_dogs_flat:
    sum += logistic_fun(np.matmul(np.array([dog]), sess.run(W)) + sess.run(b))[0][0]
print("Prediction that it is dog on dogs pictures: ", 1 - (sum / TOTAL_DOGS_TEST))


