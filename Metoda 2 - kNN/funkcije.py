import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical

import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.data import Dataset

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def trim(df, max_samples, min_samples, column):
    df=df.copy()
    groups=df.groupby(column)    
    trimmed_df = pd.DataFrame(columns = df.columns)
    groups=df.groupby(column)
    for label in df[column].unique(): 
        group=groups.get_group(label)
        count=len(group)    
        if count > max_samples:
            sampled_group=group.sample(n=max_samples, random_state=123,axis=0)
            trimmed_df=pd.concat([trimmed_df, sampled_group], axis=0)
        else:
            if count>=min_samples:
                sampled_group=group        
                trimmed_df=pd.concat([trimmed_df, sampled_group], axis=0)
    print('after trimming, the maximum samples in any class is now ',max_samples, ' and the minimum samples in any class is ', min_samples)
    return trimmed_df

def show_image_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen)

    plt.figure(figsize = (20, 20))
    length = len(labels)

    if length<25:   #show maximum of 25 images
        r=length
    else:
        r=25

    for i in range(r):        
        plt.subplot(5, 5, i + 1)
        image=images[i] /255       
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()

def load_dataset(train_path, test_path, max_samples, min_samples, img_size, batch_size,
                horizontal_flip=True, rotation_range=25, width_shift_range=.25,
                            height_shift_range=.3, zoom_range=.4, show_images=True):
    train_path = train_path
    test_path = test_path

    for d in [train_path, test_path]:
        filepaths = []
        labels = []
        classlist = sorted(os.listdir(d))

        for klass in classlist:        
            classpath = os.path.join(d, klass)
            flist = sorted(os.listdir(classpath))

            for f in flist:
                fpath = os.path.join(classpath,f)
                filepaths.append(fpath)
                labels.append(klass)

        Fseries = pd.Series(filepaths, name='filepaths')
        Lseries = pd.Series(labels, name='labels') 

        if d == train_path:            
            df = pd.concat([Fseries, Lseries], axis=1)
        else:
            test_df = pd.concat([Fseries, Lseries], axis=1)

    train_df, valid_df = train_test_split(df, train_size=.9, shuffle=True, random_state=123, stratify=df['labels']) 

    # get the number of classes and the images count for each class in train_df
    classes = sorted(list(train_df['labels'].unique()))
    class_count = len(classes)

    groups = train_df.groupby('labels')


    countlist = []
    classlist = []

    for label in sorted(list(train_df['labels'].unique())):
        group = groups.get_group(label)
        countlist.append(len(group))
        classlist.append(label)

    # get the classes with the minimum and maximum number of train images
    max_value = np.max(countlist)
    max_index = countlist.index(max_value)
    max_class = classlist[max_index]
    min_value = np.min(countlist)
    min_index = countlist.index(min_value)
    min_class = classlist[min_index]


    # lets get the average height and width of a sample of the train images
    ht = 0
    wt = 0

    # select 100 random samples of train_df
    train_df_sample = train_df.sample(n=100, random_state=123,axis=0)

    for i in range (len(train_df_sample)):
        fpath = train_df_sample['filepaths'].iloc[i]
        img = plt.imread(fpath)
        shape = img.shape
        ht += shape[0]
        wt += shape[1]
        
        
        
    max_samples = max_samples
    min_samples = min_samples
    column='labels'
    train_df= trim(train_df, max_samples, min_samples, column)  
        
        

    working_dir=r'./'
    img_size = img_size
    batch_size = batch_size

    trgen = ImageDataGenerator(horizontal_flip=horizontal_flip, rotation_range=rotation_range, width_shift_range=width_shift_range,
                                height_shift_range=height_shift_range, zoom_range=zoom_range)
    t_and_v_gen = ImageDataGenerator()

    msg = '{0:70s} for train generator'.format(' ')
    print(msg, '\r', end = '') # prints over on the same line

    train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                        class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

    msg = '{0:70s} for valid generator'.format(' ')
    print(msg, '\r', end='') # prints over on the same line

    valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                        class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)


    # for the test_gen we want to calculate the batch size and test steps such that batch_size X test_steps= number of samples in test set
    # this insures that we go through all the sample in the test set exactly once.
    length = len(test_df)
    test_batch_size = sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]  
    test_steps = int(length/test_batch_size)

    msg = '{0:70s} for test generator'.format(' ')
    print(msg, '\r', end='') # prints over on the same line

    test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                        class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

    # from the generator we can get information we will need later
    classes = list(train_gen.class_indices.keys())
    class_indices = list(train_gen.class_indices.values())
    class_count = len(classes)
    labels = test_gen.labels

    print ( 'test batch size: ' ,test_batch_size, '  test steps: ', test_steps, ' number of classes : ', class_count)

    if show_images:
        show_image_samples(train_gen)

    return train_gen, test_gen, valid_gen

import functools

def combine_dims(a, i=0, n=1):
    """
    Combines dimensions of numpy array `a`, 
    starting at index `i`,
    and combining `n` dimensions
    """
    s = list(a.shape)
    combined = functools.reduce(lambda x,y: x*y, s[i:i+n+1])
    return np.reshape(a, s[:i] + [combined] + s[i+n+1:])

def show_image_samples_with_landmarks(gen):

    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen)

    plt.figure(figsize = (20, 20))
    length = len(labels)

    if length<25:   #show maximum of 25 images
        r=length
    else:
        r=25

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands = 1, static_image_mode=True, min_detection_confidence = 0.3)
    mpDraw = mp.solutions.drawing_utils

    for i in range(r):        
        plt.subplot(5, 5, i + 1)
        image=images[i].astype('uint8')
        
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

        
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()

def zapisi_rezultate(METOD_NUM, EKSP_NUM, acc, loss, val_acc, val_loss):

    METOD_NUM = METOD_NUM
    EKSP_NUM = EKSP_NUM

    txt_file_path = r"./Merenja/eksp_" + str(EKSP_NUM) + "/"
    files = [(acc, 'acc.txt'), (loss, 'loss.txt'), (val_acc, 'val_acc.txt'), (val_loss, 'val_loss.txt')]

    for data, file_name in files:

        textfile = open(txt_file_path + file_name, "w")

        #Brise sve iz fajla
        with textfile as file:
            pass

        textfile = open(txt_file_path + file_name, "w")

        for element in data:
            textfile.write(str(element) + file_name + "\n")
        textfile.close()

    subject = 'model_eksp_' + str(EKSP_NUM) 
    save_id = subject + '.h5' 
    model_save_loc = os.path.join(txt_file_path, save_id)
    model.save(model_save_loc)
    print ('model was saved as ' , model_save_loc ) 

def plot_loss_and_acc(history):

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'y', label = 'Training loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc_is_called = list(history.history.keys())[1]

    acc = history.history[acc_is_called]
    val_acc = history.history['val_' + acc_is_called]

    plt.plot(epochs, acc, 'y', label = 'Training acc')
    plt.plot(epochs, val_acc, 'r', label = 'Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()

def load_keypoint_database(data_path):
    train_x = np.load(data_path + 'train/train_x.npy')
    train_y = np.load(data_path + 'train/train_y.npy')
    
    test_x = np.load(data_path + 'test/test_x.npy')
    test_y = np.load(data_path + 'test/test_y.npy')
    
    valid_x = np.load(data_path + 'valid/valid_x.npy')
    valid_y = np.load(data_path + 'valid/valid_y.npy')
    
    return train_x, train_y, test_x, test_y, valid_x, valid_y

def save_keypoint_dataset(data_path, train_x, train_y, test_x, test_y, valid_x, valid_y):

        np.save(data_path + 'train/train_x', train_x)
        np.save(data_path + 'train/train_y', train_y)
        
        np.save(data_path + 'test/test_x', test_x)
        np.save(data_path + 'test/test_y', test_y)
        
        np.save(data_path + 'valid/valid_x', valid_x)
        np.save(data_path + 'valid/valid_y', valid_y)

def show_heatmap(model, test_x, test_y):
    
    pred = model.predict(test_x)
    
    length = pred.shape[0]

    pred_s = np.zeros(length)
    test_s = np.zeros(length)

    for i in range(length):
        pred_s[i] = np.where(pred[i] == max(pred[i]))[0][0]
        test_s[i] = np.where(test_y[i] == max(test_y[i]))[0][0]
        
    cm = tf.math.confusion_matrix(test_s, pred_s, NUM_OF_CLASSES)
    
    
    classes = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    class_count = len(classes)

    plt.figure(figsize=(12, 8))
    sns.heatmap(np.transpose(cm), annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)       
    plt.xticks(np.arange(class_count)+.5, classes, rotation=90)
    plt.yticks(np.arange(class_count)+.5, classes, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def load_dataset_wo_valid(train_path, test_path, max_samples, min_samples, img_size, batch_size,
                    horizontal_flip=True, rotation_range=25, width_shift_range=.25,
                                height_shift_range=.3, zoom_range=.4, show_images=True):
    train_path = train_path
    test_path = test_path

    for d in [train_path, test_path]:
        filepaths = []
        labels = []
        classlist = sorted(os.listdir(d))

        for klass in classlist:        
            classpath = os.path.join(d, klass)
            flist = sorted(os.listdir(classpath))

            for f in flist:
                fpath = os.path.join(classpath,f)
                filepaths.append(fpath)
                labels.append(klass)

        Fseries = pd.Series(filepaths, name='filepaths')
        Lseries = pd.Series(labels, name='labels') 

        if d == train_path:            
            df = pd.concat([Fseries, Lseries], axis=1)
        else:
            test_df = pd.concat([Fseries, Lseries], axis=1)

    train_df = df 

    # get the number of classes and the images count for each class in train_df
    classes = sorted(list(train_df['labels'].unique()))
    class_count = len(classes)
    
    groups = train_df.groupby('labels')

    countlist = []
    classlist = []

    for label in sorted(list(train_df['labels'].unique())):
        group = groups.get_group(label)
        countlist.append(len(group))
        classlist.append(label)

    # get the classes with the minimum and maximum number of train images
    max_value = np.max(countlist)
    max_index = countlist.index(max_value)
    max_class = classlist[max_index]
    min_value = np.min(countlist)
    min_index = countlist.index(min_value)
    min_class = classlist[min_index]


    # lets get the average height and width of a sample of the train images
    ht = 0
    wt = 0

    # select 100 random samples of train_df
    train_df_sample = train_df.sample(n=100, random_state=123,axis=0)

    for i in range (len(train_df_sample)):
        fpath = train_df_sample['filepaths'].iloc[i]
        img = plt.imread(fpath)
        shape = img.shape
        ht += shape[0]
        wt += shape[1]
        
        
        
    max_samples = max_samples
    min_samples = min_samples
    column='labels'
    train_df= trim(train_df, max_samples, min_samples, column)  
        
        
    
    working_dir=r'./'
    img_size = img_size
    batch_size = batch_size

    trgen = ImageDataGenerator(horizontal_flip=horizontal_flip, rotation_range=rotation_range, width_shift_range=width_shift_range,
                                height_shift_range=height_shift_range, zoom_range=zoom_range)
    t_and_v_gen = ImageDataGenerator()

    msg = '{0:70s} for train generator'.format(' ')
    print(msg, '\r', end = '') # prints over on the same line
    
    
    
    train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

    ##msg = '{0:70s} for valid generator'.format(' ')
    print(msg, '\r', end='') # prints over on the same line

#     valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
#                                        class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)


    # for the test_gen we want to calculate the batch size and test steps such that batch_size X test_steps= number of samples in test set
    # this insures that we go through all the sample in the test set exactly once.
    length = len(test_df)
    test_batch_size = sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]  
    test_steps = int(length/test_batch_size)

    msg = '{0:70s} for test generator'.format(' ')
    print(msg, '\r', end='') # prints over on the same line
    
    print("pusi kurac")
    print(test_df)
    
    #test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col= 'labels', target_size=img_size,
                                       #class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

    # from the generator we can get information we will need later
    classes = list(train_gen.class_indices.keys())
    class_indices = list(train_gen.class_indices.values())
    class_count = len(classes)
    labels = test_gen.labels

    print ( 'test batch size: ' ,test_batch_size, '  test steps: ', test_steps, ' number of classes : ', class_count)
    
    if show_images:
        show_image_samples(train_gen)
    
    return train_gen, test_gen#, valid_gen

def load_single_dataset(df_path, max_samples, min_samples, img_size, batch_size):
    
    train_path = df_path
    
    d = train_path
    
    filepaths = []
    labels = []
    classlist = sorted(os.listdir(d))

    for klass in classlist:        
        classpath = os.path.join(d, klass)
        flist = sorted(os.listdir(classpath))

        for f in flist:
            fpath = os.path.join(classpath,f)
            filepaths.append(fpath)
            labels.append(klass)

    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    
    train_df = pd.concat([Fseries, Lseries], axis=1)
    
    classes = sorted(list(train_df['labels'].unique()))
    class_count = len(classes)
    
    groups = train_df.groupby('labels')
    
    countlist = []
    classlist = []
    
    for label in sorted(list(train_df['labels'].unique())):
        group = groups.get_group(label)
        countlist.append(len(group))
        classlist.append(label)
        
    # get the classes with the minimum and maximum number of train images
    max_value = np.max(countlist)
    max_index = countlist.index(max_value)
    max_class = classlist[max_index]
    min_value = np.min(countlist)
    min_index = countlist.index(min_value)
    min_class = classlist[min_index]
    
    # lets get the average height and width of a sample of the train images
    ht = 0
    wt = 0

    # select 100 random samples of train_df
    train_df_sample = train_df.sample(n=100, random_state=123,axis=0)

    for i in range (len(train_df_sample)):
        fpath = train_df_sample['filepaths'].iloc[i]
        img = plt.imread(fpath)
        shape = img.shape
        ht += shape[0]
        wt += shape[1]    
        
    max_samples = max_samples
    min_samples = min_samples
    column='labels'
    train_df= trim(train_df, max_samples, min_samples, column) 
    
    trgen = ImageDataGenerator()
    
    train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
    
    classes = list(train_gen.class_indices.keys())
    class_indices = list(train_gen.class_indices.values())
    class_count = len(classes)
    labels = train_gen.labels
    
    return train_gen

    
    

def hand_border(points, buffer):
    
    x = points[:, 0]
    y = points[:, 1]
    
    min_x = nn(int(np.min(x) - buffer))
    max_x = nn(int(np.max(x) + buffer))

    min_y = nn(int(np.min(y) - buffer))
    max_y = nn(int(np.max(y) + buffer))
    
    return [min_x, max_x, min_y, max_y]


def cut_n_resize(img, border, dimensions):
    
    #print("border = ", border)
    cropped_img = img[border[2]:border[3], border[0]:border[1]]

    resized_img = cv2.resize(cropped_img, dimensions)
    
    return resized_img

def nn(a, upper_treshold=np.inf):
  if a < 0:
    return 0
  if a >= upper_treshold:
    return upper_treshold - 1
  return a

def find_central_color_range(img, keypoints):
    
    lower_range = np.array([255, 255, 255])
    upper_range = np.array([0, 0, 0])
    
    for i in range(21):
        for j in range(3):    
            x = nn(int(keypoints[i, 0]), img.shape[0])
            y = nn(int(keypoints[i, 1]), img.shape[1])
            #print(lower_range[j])
            lower_range[j] = min(img[y, x, j], lower_range[j])
            upper_range[j] = max(img[y, x, j], upper_range[j])
            
    return lower_range, upper_range


def show_binarised_image_samples(gen,
                                 buffer,
                                 dimensions = (128, 128),
                                 lower_range_tightness = 0,
                                 upper_range_tightness = 0):
    
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen)
    
    plt.figure(figsize = (20, 20))
    length = len(labels)
    
    if length<25:   #show maximum of 25 images
        r=length
    else:
        r=25
    
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands = 1, static_image_mode=True, min_detection_confidence = 0.1)
    
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image=images[i].astype('uint8')
            
        mask = get_hand_mask(image,
                              buffer,
                              dimensions,
                              lower_range_tightness,
                              upper_range_tightness,
                            hands)
        
        plt.imshow(mask)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    plt.show()



def generate_kNN_database(generator,
                          num_of_samples,
                          buffer = 20,
                          dimensions = (200, 200),
                          lower_range_tightness = 0,
                          upper_range_tightness = 0,
                         num_of_rows = 10):
    
    
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands = 1, static_image_mode=True, min_detection_confidence = 0.3)
    
    data = np.zeros((num_of_samples * 24, num_of_rows * num_of_rows))
    lb_count = np.zeros(24)
    labels = []
    
    samples = 0
    sector_len = dimensions[0] // num_of_rows
    
    for batch in generator:
        for i in range(batch[0].shape[0]):
            
            image = batch[0][i].astype('uint8')
            label = batch[1][i] 
            label_name = np.where(label == max(label))[0][0]
            
            if lb_count[label_name] >= num_of_samples:
                continue
            
            mask = get_hand_mask(image,
                                  buffer,
                                  dimensions,
                                  lower_range_tightness,
                                  upper_range_tightness,
                                hands)
            for i in range (num_of_rows):
                for j in range(num_of_rows):
                    data[samples][i * num_of_rows + j] = np.sum(mask[i*sector_len : (i+1)*sector_len, j*sector_len : (j+1)*sector_len]) / (sector_len**2 * 255)
                    
            labels.append(label)
            lb_count[label_name] += 1
            samples += 1
            message = str(samples)
            sys.stdout.write('\r'+ "Generated  " + message + "  samples out of  " + str(num_of_samples * 24) + "  samples")

            if samples >= num_of_samples * 24:
                break
                
        if samples >= num_of_samples * 24:
                break
    
    sys.stdout.write('\r'+ "DONE: Generated  " + message + "  samples")
    exit_labels = np.array(labels)
    
    return data, exit_labels     



def organise_hand_keypoints(img, results):
    
    data = np.zeros((21, 2))
    img_shape = img.shape
    
    for handLms in results.multi_hand_landmarks:   
        for id, lm in enumerate(handLms.landmark):
            data[id, 0] = lm.x * img_shape[0]
            data[id, 1] = lm.y * img_shape[1]
            
    return data           



def get_hand_mask(image,
                  buffer,
                  dimensions = (128, 128),
                  lower_range_tightness = 0,
                  upper_range_tightness = 0,
                 hands = mp.solutions.hands.Hands(max_num_hands = 1, static_image_mode=True, min_detection_confidence = 0.1)):

    image_cut = np.zeros((128, 128, 3))
    mask = image_cut.copy()

    results = hands.process(image)
    if results.multi_hand_landmarks:

        keypoints = organise_hand_keypoints(image, results)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        #color = central_color(image, keypoints)
        lower_range, upper_range = find_central_color_range(image_hsv, keypoints)
        #print(color)

        border = hand_border(keypoints, buffer)
        image_cut = cut_n_resize(image, border, dimensions)
        image_cut_hsv = cv2.cvtColor(image_cut, cv2.COLOR_RGB2HSV)

        for j in range(len(lower_range)):
            lower_range[j] *= (1 +  lower_range_tightness)
            upper_range[j] *= (1 - upper_range_tightness)

        mask = cv2.inRange(np.array(image_cut_hsv), lower_range, upper_range)

        kernel1 = np.ones((5, 5), np.uint8) 
        kernel2 = np.ones((3, 3), np.uint8) 

        mask = cv2.erode(mask, kernel1, iterations = 1)
        mask = cv2.dilate(mask, kernel1, iterations = 1)

        mask = cv2.dilate(mask, kernel2, iterations = 1)
        mask = cv2.erode(mask, kernel2, iterations = 1)
        
    return mask 


     