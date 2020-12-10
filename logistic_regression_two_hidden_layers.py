import numpy as np

import glob
import tensorflow as tf
import cv2

TOTAL_DOGS_TRAIN = 1199
TOTAL_CATS_TRAIN = 1149
TOTAL_PIXELS_PIC = 65536
TOTAL_DOGS_TEST = 159
TOTAL_CATS_TEST = 149


def preparing_data_from_trainSet():
    train_dogs = glob.glob("dogs vs cats/training_set/dogs/*.jpg")
    train_cats = glob.glob("dogs vs cats/training_set/cats/*.jpg")
    dogs_x1 = np.array([[np.array(cv2.cvtColor(cv2.imread(dogs), cv2.COLOR_RGB2GRAY))] for dogs in train_dogs])
    cats_x2 = np.array([[np.array(cv2.cvtColor(cv2.imread(cats), cv2.COLOR_RGB2GRAY))] for cats in train_cats])

    dogs_x1_train = np.array([np.array(mat.ravel()) for mat in dogs_x1])
    cats_x2_train = np.array([np.array(mat.ravel()) for mat in cats_x2])

    dogs_x1_train = np.array([a / 255. for a in dogs_x1_train])
    cats_x2_train = np.array([a / 255. for a in cats_x2_train])

    data_x_tr = np.concatenate([dogs_x1_train, cats_x2_train])

    return data_x_tr, dogs_x1_train, cats_x2_train


# making label dog and cat as binary array
def making_label_y():
    dogs = np.array([[1] for x in range(TOTAL_DOGS_TRAIN)])  # making every dog picture the number 1 in array
    cats = np.array([[0] for y in range(TOTAL_CATS_TRAIN)])  # making every cat picture the number 0 for in array
    label_y = np.concatenate([dogs, cats]) # merge together in same array
    return label_y


def train_result():
    cat_prediction = np.average(y.eval(session=sess, feed_dict = {x :data_cat_train}))
    print("Prediction train cat image: ", cat_prediction)
    train_error_cat = 1 - cat_prediction

    dog_prediction = np.average(y.eval(session=sess, feed_dict= {x :data_dog_train}))
    print("Prediction train dog image: ", dog_prediction)
    train_error_dog = dog_prediction
    total_train_error = (train_error_cat + train_error_dog) / 2.
    print("Train error: ", total_train_error)


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


def test_result():
    (classify_dogRight, classify_dogWrong, classify_catRight, classify_catWrong) = (0,0,0,0)
    dog_predictions = y.eval(session=sess, feed_dict= {x :data_dog_test})
    #print("dogPredictio: ", dog_predictions)
    for dogPrediction in dog_predictions:
        # There is a 0.5 classification threshold
        if dogPrediction < 0.5: # dog classify predicted if probability < 0.5
            classify_dogRight += 1
        else:
            classify_catWrong += 1

    dog_testPrediction = np.average(dog_predictions)
    print("Prediction Dog Test: ", dog_testPrediction)

    cat_predictions = y.eval(session=sess, feed_dict= {x :data_cat_test})
    #print("catPredictio: ", cat_predictions)
    for catPrediction in cat_predictions:
        if catPrediction > 0.5: # cat classify predicted if probability > 0.5
            classify_catRight += 1
        else:
            classify_dogWrong += 1

    cat_testPrediction = np.average(cat_predictions)
    print("Prediction Cat Test: ", cat_testPrediction)

    accuracy = (classify_catRight + classify_dogRight) / (TOTAL_CATS_TEST + TOTAL_CATS_TEST) # how often the classify is correct
    recall = classify_catRight / TOTAL_CATS_TEST
    precision = classify_catRight / (classify_catRight + classify_catWrong)
    test_error = (1 - dog_testPrediction + cat_testPrediction) / 2. # how often the classify is incorrect

    print("Test Error: %.4f\n" % test_error)
    print("Accuracy: %.4f" % accuracy)
    print("Recall: %.4f" % recall)
    print("Precision: %.4f" % precision)


(hidden1_size, hidden2_size) = (100, 50)
features = TOTAL_PIXELS_PIC  # number of pixels in each picture 256X256
eps = 1e-12
x = tf.compat.v1.placeholder(tf.float32, [None, features])
y_ = tf.compat.v1.placeholder(tf.float32, [None, 1])
W1 = tf.Variable(tf.random.truncated_normal([features, hidden1_size], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
z1 = tf.nn.relu(tf.matmul(x, W1)+b1)
W2 = tf.Variable(tf.random.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))
z2 = tf.nn.relu(tf.matmul(z1,W2)+b2)
W3 = tf.Variable(tf.random.truncated_normal([hidden2_size, 1], stddev=0.1))
b3 = tf.Variable(0.)
z3 = tf.matmul(z2, W3) + b3

y = 1 / (1.0 + tf.exp(-z3))

loss1 = -(y_ * tf.math.log(y + eps) + (1 - y_) * tf.math.log(1 - y + eps))
loss = tf.reduce_mean(loss1)
update = tf.compat.v1.train.GradientDescentOptimizer(0.00001).minimize(loss)

(data_x,data_dog_train, data_cat_train) = preparing_data_from_trainSet()
label = making_label_y()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(0, 10):
    sess.run(update, feed_dict={x: data_x, y_: label})


train_result()

(data_dog_test, data_cat_test) = preparing_data_testSet()

test_result()



