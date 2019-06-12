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
        self.image_paths_placeholder=tf.placeholder(tf.string,shape=(None,1),name='image_paths')
        self.labels_placeholder=tf.placeholder(tf.int32,shape=(None,1),name='label')
        self.keep_probability_placeholder=tf.placeholder(tf.float32,name='keep_probability')

        # create batches
        self.image_batch, self.label_batch = self.start_queue()
        
        
    def load_data(self):

        # Load data
        filenames = os.listdir(self.opt.data_path)
        data = np.array(["./data/train_imgs/"+filename for filename in filenames])
        labels = np.array([re.match(r"^(.*)-(.*)", filename.split('.')[0])[1] for filename in filenames])
        self.num_classes = len(set(labels))

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]

        # Split into train and validation sets
        train_data = data[:int(self.opt.data_split_ratio*len(data))]
        train_labels = labels[:int(self.opt.data_split_ratio*len(data))]
        val_data = data[int(self.opt.data_split_ratio*len(data)):]
        val_labels = data[int(self.opt.data_split_ratio*len(data)):]

        return train_data, train_labels, val_data, val_labels
    
    def start_queue(self):
        
        # Create index producer
        index_queue=tf.train.range_input_producer(self.num_data,num_epochs=None,
                                                 shuffle=True,seed=None,capacity=32)
        self.index_dequeue_op=index_queue.dequeue_many(self.opt.batch_size*self.num_batches,'index_dequeue')

        # Create data queue
        input_queue=tf.FIFOQueue(capacity=2000000,
                                 dtypes=[tf.string,tf.int32],
                                 shapes=[(1,),(1,)],
                                 shared_name=None,name=None)
        self.enqueue_op=input_queue.enqueue_many([self.image_paths_placeholder,self.labels_placeholder],
                                           name='enqueue_op')
        
        # Create batches
        image_and_labels_list=[]
        for _ in range(self.opt.threads):
            filenames, labels = input_queue.dequeue()
            images=[]
            for filename in tf.unstack(filenames):
                image_contents = tf.read_file(filename)
                image = tf.image.decode_image(image_contents, 3)
                images.append(self.image_var(image))
            image_and_labels_list.append([images,labels])
        image_batch,label_batch=tf.train.batch_join(image_and_labels_list,
                                  batch_size=self.batch_size_placeholder,
                                  shapes=[self.opt.img_size+(3,),()],
                                  enqueue_many=True,
                                  capacity=4*self.opt.threads*100,
                                  allow_smaller_final_batch=True)
        return tf.identity(image_batch), tf.identity(label_batch)
    
    def image_var(self, image):

        def random_rotate_image(image):
            angle = np.random.uniform(low=-10.0, high=10.0)
            return misc.imrotate(image, angle, 'bicubic')

        # Random flip
        image=tf.cond(tf.constant(np.random.uniform()>0.8),
                    lambda:tf.py_func(random_rotate_image,[image],tf.uint8),
                    lambda:tf.identity(image))
        # Random crop
        image=tf.cond(tf.constant(np.random.uniform()>0.5),
                        lambda:tf.random_crop(image,self.opt.img_size+(3,)),
                        lambda:tf.image.resize_image_with_crop_or_pad(image,self.opt.img_size[0],self.opt.img_size[1]))
        # Random contrast
        image=tf.cond(tf.constant(np.random.uniform()>0.7),
                        lambda:tf.image.random_contrast(image,lower=0.3, upper=1.0),
                        lambda:tf.identity(image))
        # Normalize into [-1,1]
        image=tf.cast(image,tf.float32)-127.5/128.0
        image.set_shape(self.opt.img_size+(3,))
        
        return image


    # def split_batches(self, data, labels):

    #     # Split batches
    #     self.self.num_batch = int(len(labels) / opt.batch_size)
    #     data = data[:self.num_batch * opt.batch_size]
    #     labels = labels[:self.num_batch * opt.batch_size]
    #     data = np.split(data, self.num_batch, 0)
    #     labels = np.split(labels, self.num_batch, 0)

    #     return data, labels
    
    # def next_batch(self):
    #     ret = self.data[self.pointer], self.labels[self.pointer]
    #     self.pointer = (self.pointer + 1) % self.num_batch
    #     return ret

    # def reset_pointer(self):
    #     self.pointer = 0

    # Load pickle files
    @staticmethod
    def load_pickle(file):
        with open(file,'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_pickle(data, file):
        with open(file,'wb') as f:
            pickle.dump(data, f)
