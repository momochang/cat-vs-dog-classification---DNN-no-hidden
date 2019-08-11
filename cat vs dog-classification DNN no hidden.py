import gc
import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as image
import random
import time


def gen_label(img):
    try:
        label = img.split('.')[-3]
    except:
        label = None
    finally:
        if label == 'cat':
            return [1, 0]
        elif label == 'dog':
            return [0, 1]
        else:
            return [0, 0]

def read_directory(**kwargs):

    for filename in os.listdir(kwargs['directory_name']):
        #if (filename == '')
        label = gen_label(filename)
        img = cv2.imread(kwargs['directory_name'] + "/" + filename)
        if img is None:
            pass
        else:
            kwargs['array_image'].append([img, np.array(label)])

    return np.array(kwargs['array_image'])


def display_one(a, title1 = "Original"):
    plt.imshow(a)
    plt.title(title1)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121)
    plt.imshow(a)
    plt.title(title1)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(b)
    plt.title(title2)
    plt.show()


def time_sleep():
    for i in range(5):
        time.sleep(1)
        print('.', end = '')
    print()

#release memory space 
def release_memory():
    array_of_img.clear()
    test_of_img.clear()
    datasets = None
    test_datasets = None
    return datasets, test_datasets


def print_sizeofmemory():
    print(gc.collect())

## 1. read datas that each data has three-dimension (height, width, RGB) 
#Global variable
array_of_img = []
test_of_img = []

print('----------------------------')
print('cats train_data loading', end = '')
data = read_directory(array_image = array_of_img, directory_name = 'D:/deep-learning/images/dataset/training_set/cats')
time_sleep()
print('----------------------------')
print('dogs train_data loading', end = '')
datasets = read_directory(array_image = array_of_img, directory_name = 'D:/deep-learning/images/dataset/training_set/dogs')
time_sleep()
print('----------------------------')
print('cats test_data loading', end = '')
test_data = read_directory(array_image = test_of_img, directory_name = 'D:/deep-learning/images/dataset/test_set/cats')
time_sleep()
print('----------------------------')
print('dogs test_data loading', end = '')
test_datasets = read_directory(array_image = test_of_img, directory_name = 'D:/deep-learning/images/dataset/test_set/dogs')
time_sleep()
print('----------------------------')
print('clean memeory space')



#data = np.reshape(data, (data.shape[0], data[0][0].shape[0], data[0][0].shape[1], data[0][0].shape[2]))

type(data)
print(data.shape)
#print(data[0][0].shape)

random.shuffle(datasets)
random.shuffle(test_datasets)
datasets.shape
test_datasets.shape


## 3. distribution train, valid and test data
train_image, train_label = (datasets[:, 0], datasets[:, 1])
valid_image, valid_label = (test_datasets[:1000, 0], test_datasets[:1000, 1])
test_image, test_label = (test_datasets[1000:, 0], test_datasets[1000:, 1])
print('train_image shape: ',train_image.shape)
print('valid_image shape: ',valid_image.shape)
print('test_image shape: ',test_image.shape)
print('train_label shape: ',train_label.shape)
print('valid_label shape: ',valid_label.shape)
print('test_label shape: ',test_label.shape)

datasets, test_datasets = release_memory()

## 2. resize each data
def processing(data, lab):
    #img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data[:3]]
    
    print('The Second Original size', data[1].shape)
    height = 50
    width = 50
    dim = (width, height)
    res_img = []
    res_lab = []
    for i in range(len(data)):
        res = cv2.resize(data[i], dim, interpolation = cv2.INTER_LINEAR)
        res_img.append(res)
        res_lab.append(lab[i])
        
    print("The Second Resized size", res_img[1].shape)
    resized = res_img[1]
    display(data[1], resized, 'Originial', 'Resized')
    return np.array(res_img), np.array(res_lab)


def normalization(img):
    img = img / 255.
    return img

train_image, train_label = processing(train_image, train_label)
valid_image, valid_label = processing(valid_image, valid_label)
test_image, test_label = processing(test_image, test_label)

train_image = normalization(train_image)
valid_image = normalization(valid_image)
test_image = normalization(test_image)


print('train_image shape: ', train_image.shape)
print('train_label shape: ', train_label.shape)
print('train_label the first shape: ', train_label[0].shape)
print('train_label the first type: ', type(train_label[0]))
print('train_label the first content: ', train_label[0])


## 4. initial variable
weights = None
biases = None
n_hidden1 = 3000
#n_hidden2 = 1800
alpha = 0.3
dropoout_ratio = [0.2, 0.1]
train_image = np.reshape(train_image, (train_image.shape[0], 50 * 50 * 3))
valid_image = np.reshape(valid_image, (valid_image.shape[0], 50 * 50 * 3))
test_image = np.reshape(test_image, (test_image.shape[0], 50 * 50 * 3))
n_images = 50 * 50 *3
#n_images = [50, 50, 3]
n_labels = 2
learning_rate = 0.001
graph = tf.Graph()
sess = tf.Session(graph = graph)



def structure(images, labels, weights, biases, train = False):
    if (weights is None) or (biases is None):
        weights = {
            #'conv1' : tf.Variable(tf.truncated_normal([50, 50, 3, 6], stddev = 0.1)),
            'fc1' : tf.Variable(tf.truncated_normal([n_images, n_hidden1])),
            'fc2' : tf.Variable(tf.truncated_normal([n_hidden1, n_labels])),
            #'fc3' : tf.Variable(tf.truncated_normal([n_hidden2, n_labels]))
        }
        biases = {
            #'conv1' : tf.Variable(tf.zeros([6], dtype = tf.float32)),
            'fc1' : tf.Variable(tf.zeros([n_hidden1], dtype = tf.float32)),
            'fc2' : tf.Variable(tf.zeros([n_labels], dtype = tf.float32)),
            #'fc3' : tf.Variable(tf.zeros([n_labels], dtype = tf.float32))
        }
    '''
    conv1 = get_conv_2d_layer(images,
                              weights['conv1'],
                              biases['conv1'],
                              activation = tf.nn.relu)
    pool2 = tf.nn.max_pool(conv1,
                           ksize = [1, 2, 2, 3], strides = [1, 2, 2, 3], padding = 'VALID')
    
    flatten = get_flatten_layer(pool2)
    '''
    f1 = get_dense_layer(images = images,
                          weight = weights['fc1'],
                          bias = biases['fc1'],
                          activation = tf.nn.tanh)
    
    if train:
        f1 = tf.nn.dropout(f1, keep_prob = 1 - dropoout_ratio[1])
        
    logits = get_dense_layer(images = f1,
                         weight = weights['fc2'],
                         bias = biases['fc2'],)
    '''                     activation = tf.nn.relu)
    if train:
        f2 = tf.nn.dropout(f2, keep_prob = 1 - dropoout_ratio[1])
    
    logits = get_dense_layer(images = f2,
                             weight = weights['fc3'],
                             bias = biases['fc3'])
    '''
    _y = tf.nn.softmax(logits)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))
    
    return (weights, biases, _y, loss)



def get_conv_2d_layer(images, weight, bias, strides = (1, 1), padding = 'VALID', activation = None):
    x = tf.add(
        tf.nn.conv2d(images,
                     weight,
                     [1, strides[0], strides[1], 1],
                     padding = padding), bias)
    if activation:
        x = activation(x)
    return x



def get_dense_layer(images, weight, bias, activation = None):
    x = tf.add(tf.matmul(images, weight), bias)
    if activation:
        x = activation(x)
    return x



def get_flatten_layer(input_layer):
    shape = input_layer.get_shape().as_list() #only tensor can used
    n = 1
    for s in shape[1:]:
        n = n * s
        print('n :', n)
    x = tf.reshape(input_layer, [-1, n])
    return x


## 5. initial Graph structure
with graph.as_default():
    train_images = tf.placeholder(tf.float32, [None, n_images])
    train_labels = tf.placeholder(tf.float32, [None, n_labels])
    
    weights, biases, _y, orig_loss = structure(images = train_images,
                                          labels = train_labels,
                                          weights = weights,
                                          biases = biases,
                                          train = True)
    
    regularization = tf.reduce_sum([tf.nn.l2_loss(w) for w in weights.values()]) \
                / tf.reduce_sum([tf.size(w, out_type = tf.float32) for w in weights.values()])
    loss = orig_loss + alpha * regularization
    
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    new_images = tf.placeholder(tf.float32, [None, n_images])
    new_labels = tf.placeholder(tf.float32, [None, n_labels])
    weights, biases, new_y, new_orig_loss = structure(images = new_images,
                                                 labels = new_labels,
                                                 weights = weights,
                                                 biases = biases)
    new_loss = new_orig_loss + alpha * regularization
    init_op = tf.global_variables_initializer()


def _check_array(ndarray):
    ndarray = np.array(ndarray)
    if len(ndarray) == 1:
        ndarray = np.reshape(ndarray, (1, ndarray.shape[0]))
    return ndarray


def predict(X):
    X = _check_array(X)
    #y = _check_array(y)
    
    feed_dict = {
         new_images : X
     #    new_labels : y
    }
    return sess.run(new_y, feed_dict = feed_dict)




def evaluate(X, y):
    X = _check_array(X)
    y = _check_array(y)
    feed_dict = {
        new_images : X,
        new_labels : y
    }
    return sess.run(new_loss, feed_dict)



def accuracy(predictions, labels):
    return np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def fit(X, y, epochs = 10, learning_rate = 1e-3, valid_data = None, test_data = None):
    X = _check_array(X)
    y = _check_array(y)
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    val_loss_list = []
    val_acc_list = []
    sess.run(init_op)
    for epoch in range(epochs):
        print('Epoch %2d/%2d: ' % (epoch + 1, epochs))
        
        feed_dict = {
            train_images : X,
            train_labels : y
        }
        sess.run(train_op, feed_dict = feed_dict)
        Y = predict(X)
        train_loss = evaluate(X, y)
        train_acc = accuracy(Y, y)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        print('train loss %8.4f, train acc %3.2f%%' % (train_loss, train_acc * 100), end = '  ')
        
        if valid_data is not None:
            val_Y = predict(valid_data[0])
            val_loss = evaluate(valid_data[0], valid_data[1])
            val_acc = accuracy(val_Y, valid_data[1])
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            print('val loss %8.4f, val acc %3.2f%%' % (val_loss, val_acc * 100))
            
        if test_data is not None:
            test_Y = predict(test_data[0])
            test_acc = accuracy(test_Y, test_data[1])
            test_acc_list.append(test_acc)
            print('test acc %3.2f%%' % (test_acc * 100))
    
    sess.close()
    
    ans = {
        'train_loss' : train_loss_list,
        'train_acc' : train_acc_list,
        'val_loss' : val_loss_list,
        'val_acc' : val_acc_list,
        'test_acc' : test_acc_list
    }
    
    return ans

dic_ans = fit(X = train_image,
              y = train_label,
              epochs = 100,
              learning_rate = learning_rate,
              valid_data = (valid_image, valid_label),
              test_data = (test_image, test_label))



plt.subplot(211)
p1, = plt.plot(dic_ans['train_loss'], label = 'train_ls')
p2, = plt.plot(dic_ans['val_loss'], label = 'valid_ls')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('model result')
plt.legend([p1, p2], loc = 'upper right')
plt.subplot(212)
plt.plot(dic_ans['train_acc'])
plt.plot(dic_ans['val_acc'])
plt.plot(dic_ans['test_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
