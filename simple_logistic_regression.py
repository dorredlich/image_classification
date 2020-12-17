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


def preparing_data_from_trainSet():
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
    cats = np.array([[1] for x in range(TOTAL_CATS_TRAIN)])  # making every dog picture the number 1 in array
    dogs = np.array([[0] for y in range(TOTAL_DOGS_TRAIN)])  # making every cat picture the number 0 for in array
    label_y = np.concatenate([cats, dogs]) # merge together in same array
    return label_y


def preparing_data_testSet():
    test_dogs = glob.glob("dogs vs cats/test_set/dogs/*.jpg")
    test_cats = glob.glob("dogs vs cats/test_set/cats/*.jpg")
    data_test_dogs = np.array([[np.array(cv2.cvtColor(cv2.imread(dog), cv2.COLOR_RGB2GRAY))] for dog in test_dogs])
    data_test_cats = np.array([[np.array(cv2.cvtColor(cv2.imread(cat), cv2.COLOR_RGB2GRAY))] for cat in test_cats])

    data_test_dogs_flat = np.array([np.array(mat.ravel()) for mat in data_test_dogs])
    data_test_cats_flat = np.array([np.array(mat.ravel()) for mat in data_test_cats])

    data_test_dogs_flat = np.array([x / 255. for x in data_test_dogs_flat])
    data_test_cats_flat = np.array([x / 255. for x in data_test_cats_flat])
    return data_test_dogs_flat,data_test_cats_flat


def train_result():
    sumCats = 0
    for cat in data_cat_train:
        sumCats += logistic_fun(np.matmul(np.array([cat]), sess.run(W)) + sess.run(b))[0][0]

    cat_prediction = sumCats / TOTAL_CATS_TRAIN
    print("Prediction in train that it is cat on cats pictures: ", cat_prediction)
    cat_error = 1 - sumCats / TOTAL_CATS_TRAIN

    sumDog = 0
    for dog in data_dog_train:
        sumDog += logistic_fun(np.matmul(np.array([dog]), sess.run(W)) + sess.run(b))[0][0]

    dog_prediction = sumDog / TOTAL_DOGS_TRAIN
    print("Prediction in train that it is dog on dogs pictures: ", dog_prediction)
    dog_error = dog_prediction
    total_train_error = (dog_error + cat_error) / 2.
    print("Total Train Error: ", total_train_error)


def test_result():
    (classify_dogRight, classify_catWrong, classify_catRight, classify_dogWrong) = (0, 0, 0, 0)
    (sumDog, sumCat) = (0, 0)
    for dog in data_dog_test:
        dogPrediction = logistic_fun(np.matmul(np.array([dog]), sess.run(W)) + sess.run(b))[0][0]
        sumDog += dogPrediction
        # There is a 0.5 classification threshold
        if dogPrediction < 0.5:  # dog classify predicted if probability < 0.5
            classify_dogRight += 1
        else:
            classify_catWrong += 1
    dog_testPrediction = sumDog / TOTAL_DOGS_TEST

    dog_test_error = dog_testPrediction

    for cat in data_cat_test:
        catPrediction = logistic_fun(np.matmul(np.array([cat]), sess.run(W)) + sess.run(b))[0][0]
        sumCat += catPrediction
        if catPrediction > 0.5:  # cat classify predicted if probability > 0.5
            classify_catRight += 1
        else:
            classify_dogWrong += 1
    cat_testPrediction = sumCat / TOTAL_CATS_TEST

    cat_test_error = 1 - cat_testPrediction

    accuracy = (classify_catRight + classify_dogRight) / (TOTAL_CATS_TEST + TOTAL_CATS_TEST)  # how often the classify is correct
    recall = classify_catRight / TOTAL_CATS_TEST
    precision = classify_catRight / (classify_catRight + classify_catWrong)
    test_error = (1 - dog_test_error + cat_test_error) / 2.  # how often the classify is incorrect

    print("Test Error: ", test_error)
    print("\nAccuracy: ",  accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)


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

(data_x,data_dog_train, data_cat_train) = preparing_data_from_trainSet()
label = making_label_y()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(0, 7000):
    sess.run(update, feed_dict={x: data_x, y_: label})

train_result()

(data_dog_test, data_cat_test) = preparing_data_testSet()

test_result()
