import tensorflow as tf
import numpy as np
import cv2
import argparse
import pickle
import os
from PIL import Image
import pandas as pd
from config import opt

# Image helper
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Image recognition helper
def recognize_pill(image_np, boxes, classes, scores, labels):

    im_width = image_np.shape[0]
    im_height = image_np.shape[1]

    image_arr = []
    flag = 0

    # Prepare detected images
    for i in range(min(opt.max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > opt.min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            
            (left, right, top, bottom) = (round(xmin * im_width), round(xmax * im_width),
                                          round(ymin * im_height), round(ymax * im_height))
            crop_image = image_np[left:right,top:bottom,:]
            scaled =cv2.resize(crop_image,(149, 149),interpolation=cv2.INTER_LINEAR)-127.5/128.0
            image_arr.append(scaled)
            flag = 1
    
    if flag == 0:
        print("Nothing detected!!")
        return np.asarray(classes).astype(np.int32), np.asarray(scores)

    image_arr = np.asarray(image_arr)

    # Load pre-extracted database embeddings
    with open(opt.feature_save_path, "rb") as f:
        feat_database = pickle.load(f)

    # Start to detect

    with tf.Graph().as_default():
        with tf.Session() as sess:

            filename = opt.model_dir+"model.ckpt-{}.meta".format(opt.restore_index)
            if os.path.exists(filename):
                saver = tf.train.import_meta_graph(filename)
                saver.restore(sess, tf.train.latest_checkpoint(opt.model_dir))    
            
            features = tf.get_default_graph().get_tensor_by_name("features:0")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("image_input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name("keep_probability:0")
            
            feed_dict = {images_placeholder: image_arr, 
            phase_train_placeholder: False,
            keep_probability_placeholder: 1.0}

            feats = sess.run(features, feed_dict=feed_dict)
            
            renew_classes = ['Others']*feats.shape[0]
            renew_scores = [0]*boxes.shape[0]
            for i in range(feats.shape[0]):
                diff=np.mean(np.square(feats[i]-feat_database),axis=1)
                print(diff)
                if min(diff)<opt.embed_threshold:
                    index=np.argmin(diff)
                    score = 1/(1+np.exp(-(1-min(diff))))
                    renew_classes[i] = labels[index]
                    renew_scores[i] = score
    
    return np.asarray(renew_classes).astype(np.int32), \
           np.asarray(renew_scores)

