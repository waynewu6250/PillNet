import numpy as np
import json
import re
import pickle
import os
import tensorflow as tf
from scipy import misc

class ImageData:

    def __init__(self, opt):

        self.opt = opt
        
        # Load image_path
        self.train_data, self.train_labels,  self.val_data,  self.val_labels = self.load_data()
        self.num_data = len(self.train_data)
        self.num_batches = self.num_data // opt.batch_size
        self.num_val_data = len(self.val_data)
        self.num_val_batches = self.num_val_data // opt.batch_size

        # placeholders
        self.batch_size_placeholder=tf.placeholder(tf.int32,name='batch_size')
        self.phase_train_placeholder=tf.placeholder(tf.bool,name='phase_train')
        self.image_paths_placeholder=tf.placeholder(tf.string,shape=(None,),name='image_paths')
        self.labels_placeholder=tf.placeholder(tf.int32,shape=(None,),name='label')
        self.keep_probability_placeholder=tf.placeholder(tf.float32,name='keep_probability')
    
    
    def class_to_id(self, labels):

        class2id = {}
        counter = 0
        for label in labels:
            if label not in class2id:
                class2id[label] = counter
                counter += 1
        id2class = {v:k for k,v in class2id.items()}
        labels = np.array([class2id[label] for label in labels])

        return labels, class2id, id2class
        
    
    def load_data(self):

        # Load data
        filenames = os.listdir(self.opt.data_path)
        data = np.array(["./data/train_imgs/"+filename for filename in filenames])
        labels = np.array([re.match(r"^(.*)-(.*)", filename.split('.')[0])[1] for filename in filenames])
        labels, self.class2id, self.id2class = self.class_to_id(labels)
        self.num_classes = len(self.class2id)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]

        # Split into train and validation sets
        train_data = data[:int(self.opt.data_split_ratio*len(data))]
        train_labels = labels[:int(self.opt.data_split_ratio*len(data))]
        val_data = data[int(self.opt.data_split_ratio*len(data)):]
        val_labels = labels[int(self.opt.data_split_ratio*len(data)):]

        return train_data, train_labels, val_data, val_labels
    
    
    def start_queue(self):

        # Create index producer (For all data)
        index_queue=tf.train.range_input_producer(self.num_data,num_epochs=None,
                                                shuffle=True,seed=None,capacity=32)
        self.index_dequeue_op=index_queue.dequeue_many(self.opt.batch_size*self.num_batches,'index_dequeue')

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths_placeholder, self.labels_placeholder))
        dataset = dataset.map(self._parse_function, num_parallel_calls=4)
        dataset = dataset.batch(self.opt.batch_size).repeat(self.opt.max_epoch)

        iterator = dataset.make_initializable_iterator()
        image_batch, label_batch = iterator.get_next()

        return iterator, tf.identity(image_batch), tf.identity(label_batch)
    
    @staticmethod
    def _parse_function(filename, label):

        img_size = (149, 149)

        def random_rotate_image(image):
            angle = np.random.uniform(low=-10.0, high=10.0)
            return misc.imrotate(image, angle, 'bicubic')
        
        # Read file
        image_contents = tf.read_file(filename)
        image = tf.image.decode_image(image_contents, channels=3)

        # Random flip
        image=tf.cond(tf.constant(np.random.uniform()>0.8),
                    lambda:tf.py_func(random_rotate_image,[image],tf.uint8),
                    lambda:tf.identity(image))
        # Random crop
        image=tf.image.resize_image_with_crop_or_pad(image,img_size[0],img_size[0])
        # Random contrast
        image=tf.cond(tf.constant(np.random.uniform()>0.7),
                        lambda:tf.image.random_contrast(image,lower=0.3, upper=1.0),
                        lambda:tf.identity(image))
        # Normalize into [-1,1]
        image=tf.cast(image,tf.float32)-127.5/128.0
        image.set_shape(img_size+(3,))
        
        return image, label

    
    # Load pickle files
    @staticmethod
    def load_pickle(file):
        with open(file,'rb') as f:
            return pickle.load(f)
    
    
    @staticmethod
    def save_pickle(data, file):
        with open(file,'wb') as f:
            pickle.dump(data, f)
