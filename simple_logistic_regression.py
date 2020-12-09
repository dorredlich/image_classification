import numpy as np

import glob
import tensorflow as tf
import cv2

TOTAL_DOGS_TRAIN = 1199
TOTAL_CATS_TRAIN = 1149
TOTAL_PIXELS_PIC = 65536
TOTAL_DOGS_TEST = 159
TOTAL_CATS_TEST = 149



def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))


def preparing_data_x_from_trainSet():
    train_dogs = glob.glob("dogs vs cats/training_set/dogs/*.jpg")
    train_cats = glob.glob("dogs vs cats/training_set/cats/*.jpg")
    dogs_x1 = np.array([[np.array(cv2.cvtColor(cv2.imread(dogs), cv2.COLOR_RGB2GRAY))] for dogs in train_dogs])
    cats_x2 = np.array([[np.array(cv2.cvtColor(cv2.imread(cats), cv2.COLOR_RGB2GRAY))] for cats in train_cats])

    dogs_x1_flatten = np.array([np.array(mat.ravel()) for mat in dogs_x1])
    cats_x2_flatten = np.array([np.array(mat.ravel()) for mat in cats_x2])

    dogs_x1_flatten = np.array([a / 255. for a in dogs_x1_flatten])
    cats_x2_flatten = np.array([a / 255. for a in cats_x2_flatten])

    data_x_tr = np.concatenate([dogs_x1_flatten, cats_x2_flatten])

    return data_x_tr, dogs_x1_flatten, cats_x2_flatten

# making label dog and cat as binary array
def making_label_y():
    dogs = np.array([[1] for x in range(TOTAL_DOGS_TRAIN)])  # making every dog picture the number 1 in array
    cats = np.array([[0] for y in range(TOTAL_CATS_TRAIN)])  # making every cat picture the number 0 for in array
    label_y = np.concatenate([dogs, cats]) # merge together in same array
    return label_y

def training_print():
    cat_prediction = np.average(y.eval(session=sess, feed_dict = 	{x :data_cat_train}))
    print("Prediction train cat image: ", cat_prediction)
    train_error_cat = 1 - cat_prediction

    dog_prediction = np.average(y.eval(session=sess, feed_dict= {x :data_dog_train}))
    print("Prediction train dog image: ", dog_prediction)
    train_error_dog = dog_prediction
    total_train_error = (train_error_cat + train_error_dog) / 2.
    print("Train error: ", total_train_error)























    


loss1 = -(y_ * tf.math.log(y + eps) + (1 - y_) * tf.math.log(1 - y + eps))
loss = tf.reduce_mean(loss1)
update = tf.compat.v1.train.GradientDescentOptimizer(0.00001).minimize(loss)
(data_x,data_dog_train, data_cat_train) = preparing_data_x_from_trainSet()
label = making_label_y()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(0, 10):
    sess.run(update, feed_dict={x: data_x, y_: label})