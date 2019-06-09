import tensorflow as tf
import os

from config import opt
from data import ImageData
from model import PillNet


def train():

    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        # Load Data
        alldata = ImageData(opt)

        # Model
        net = PillNet(alldata.image_batch,
                      alldata.label_batch,
                      alldata.num_data,
                      opt)
        
        total_loss, accuracy = net.inference(alldata.keep_probability_placeholder,
                                             alldata.phase_train_placeholder,
                                             opt.embedding_size,
                                             opt.weight_decay)
        train_op = net.optimize(total_loss, tf.trainable_variables(), global_step)

        saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=3)
        summary_op=tf.summary.merge_all()
        init = tf.global_variables_initializer()
        
        ###########################################################################
        ############               Start to run the graph              ############
        ###########################################################################

        with tf.Session() as sess:

            sess.run(init)
            train_writer=tf.summary.FileWriter('graph/train/',sess.graph)
            val_writer = tf.summary.FileWriter('graph/val/', sess.graph)
            
            # Run data batch
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            # Load model
            if os.path.exists(opt.model_dir):
                model_file = tf.train.latest_checkpoint(opt.model_dir)
                if model_file:
                    sess.restore(sess, model_file)
                    print("Load from latest checkpoint")
            else:
                os.mkdir(opt.model_dir)
            
            for epoch in range(opt.max_epoch):

                global_step = sess.run(global_step)
                

            










