import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from sklearn.model_selection import train_test_split

from Preprocessing import ImageTransformer
 

def preprocess_data_tf(im, label):

    im = tf.cast(im, tf.float32)
    im = im/127.5
    im = im - 1

    return im, label

def preprocess_data(im_tilde, label):

    im_tilde = im_tilde.astype(np.float32)
    im_tilde = im_tilde/127.5
    im_tilde = im_tilde - 1

    return im_tilde, label


def label_image(frame_name):

    if 'SW=0' in os.path.splitext(frame_name)[0]:
        return 0
    elif 'SW=1' in os.path.splitext(frame_name)[0]:
        return 1

def set_dataset(case_dir, img_dims, scan=False):

    if scan:
        samples_path = []
        for (root, case_dirs, _) in os.walk(os.path.join(case_dir,'Datasets','Datasets_orig')):
            for case_dir in case_dirs:
                for (case_root, aoa_dirs, _) in os.walk(os.path.join(root,case_dir)):
                    for aoa_dir in aoa_dirs:
                        for (aoa_root, mach_dirs, _) in os.walk(os.path.join(case_root,aoa_dir)):
                            for mach_dir in mach_dirs:
                                for (mach_root, dirs, _) in os.walk(os.path.join(aoa_root,mach_dir)):
                                    if dirs:
                                        slices_dir = dirs[0]
                                        samples = [os.path.join(mach_root,slices_dir,file) for file in
                                                   os.listdir(os.path.join(root,case_root,aoa_root,mach_root,slices_dir))]
                                        for sample in samples:
                                            samples_path.append(sample)
    else:  # If samples are directly storaged in folder
        samples_path = []
        for (root, case_dirs, _) in os.walk(os.path.join(case_dir,'Datasets','Datasets_orig')):
            for case_dir in case_dirs:
                samples = os.listdir(os.path.join(root,case_dir))
                for sample in samples:
                    samples_path.append(os.path.join(root,case_dir,sample))

    # Generate X,y datasets
    m = len(samples_path)
    X = np.zeros((m,img_dims[1],img_dims[0],3),dtype='uint8')
    y = np.zeros((m,),dtype=int)
    for i,sample in enumerate(samples_path):
        # X-array storage
        img = cv.imread(sample)
        X[i,:,:,:] = ImageTransformer.resize(img,img_dims)
        # y-label storage
        y[i] = label_image(os.path.basename(sample))

    return X, y

def standardize_image_size(X, img_dims):

    m = X.shape[0]
    X_resized = np.zeros((m,img_dims[1],img_dims[0],3),dtype='uint8')
    for i in range(m):
        if X[i].shape[0:2] != (img_dims[1],img_dims[0]):
            X_resized[i] = ImageTransformer.resize(X[i],img_dims)
        else:
            X_resized[i] = X[i]

    return X_resized

def read_preset_datasets(case_dir, dataset_ID=None):

    if dataset_ID == None:
        dataset_dir = [os.path.join(case_dir,'Dataset')]
    else:
        dataset_dir = [os.path.join(case_dir,'Dataset_{}'.format(i)) for i in dataset_ID]

    X = []
    y = []
    for folder in dataset_dir:
        if os.path.exists(folder):
            files = [os.path.join(folder,file) for file in os.listdir(folder)]
            for i,file in enumerate(files):
                X.append(cv.imread(file))
                y.append(label_image(file))
        else:
            X = None
            y = None
    X = np.array(X)
    y = np.array(y,dtype=int)

    return X, y

def get_test_dataset(case_dir, img_dims):

    # Read original datasets
    X, y = set_dataset(case_dir,img_dims)
    # Resize images, if necessary
    X = standardize_image_size(X,img_dims)

    return (X,y)

def get_datasets(case_dir, img_dims, train_size, add_augmented=False, augdataset_ID=[1]):

    # Read original datasets
    X_orig, y_orig = set_dataset(case_dir,img_dims)
    # Resize images, if necessary
    X_orig = standardize_image_size(X_orig,img_dims)

    if add_augmented:
        # Check for augmented datasets
        aug_case_dir = os.path.join(case_dir,'Datasets','Datasets_augmented')
        X_aug, y_aug = read_preset_datasets(aug_case_dir,augdataset_ID)
        X_aug = standardize_image_size(X_aug,img_dims)
        try:
            # Join both datasets
            X = np.concatenate((X_orig,X_aug),axis=0)
            y = np.concatenate((y_orig,y_aug),axis=0)
        except:
            X, y = X_orig, y_orig
    else:
        X, y = X_orig, y_orig

    X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=train_size,shuffle=True)
    X_cv, X_test, y_cv, y_test = train_test_split(X_val,y_val,train_size=0.75,shuffle=True)

    data_train = (X_train, y_train)
    data_cv = (X_cv, y_cv)
    data_test = (X_test, y_test)
    
    return data_train, data_cv, data_test

def create_dataset_pipeline(dataset, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):
    X, y = dataset
    y_oh = tf.one_hot(y,depth=1)
    dataset_tensor = tf.data.Dataset.from_tensor_slices((X, y_oh))

    if is_train:
        dataset_tensor = dataset_tensor.shuffle(buffer_size=X.shape[0]).repeat()
    dataset_tensor = dataset_tensor.map(preprocess_data_tf, num_parallel_calls=num_threads)
    dataset_tensor = dataset_tensor.batch(batch_size)
    dataset_tensor = dataset_tensor.prefetch(prefetch_buffer)

    return dataset_tensor

    
def get_tensorflow_datasets(data_train, data_cv, data_test, batch_size=32):

    dataset_train = create_dataset_pipeline(data_train,is_train=True,batch_size=batch_size)
    dataset_cv = create_dataset_pipeline(data_cv,is_train=False,batch_size=1)
    dataset_test = preprocess_data_tf(data_test[0],tf.one_hot(data_test[1],depth=1))
    
    return dataset_train, dataset_cv, dataset_test