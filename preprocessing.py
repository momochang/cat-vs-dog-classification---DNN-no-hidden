import cv2
import os
import time
import matplotlib.pyplot as plt
import gc
import numpy as np

## 1. read datas that each data has three-dimension (height, width, RGB) 
#Global variable
array_of_img = []
test_of_img = []

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
def release_memory(*args):
    del args['array_of_img']
    del args['test_of_img']
    del args['train_cat']
    del args['train_dog']
    del args['test_cat']
    del args['test_dog']
    


def print_sizeofmemory():
    print(gc.collect())
    
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



