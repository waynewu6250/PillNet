import tensorflow as tf
import os
import numpy as np

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
                      alldata.num_classes,
                      opt)

        logits, center_loss, cross_entropy_mean, total_loss, accuracy = net.inference(alldata.keep_probability_placeholder,
                                                                                      alldata.phase_train_placeholder,
                                                                                      opt.embedding_size,
                                                                                      opt.weight_decay)
        train_op = net.optimize(
            total_loss, tf.trainable_variables(), global_step)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        summary_op = tf.summary.merge_all()
        init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

        ###########################################################################
        ############               Start to run the graph              ############
        ###########################################################################

        with tf.Session() as sess:

            sess.run(init)
            train_writer = tf.summary.FileWriter('graph/train/', sess.graph)
            val_writer = tf.summary.FileWriter('graph/val/', sess.graph)

            # Run data batch
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            # Load model
            if os.path.exists(opt.model_dir):
                model_file = tf.train.latest_checkpoint(opt.model_dir)
                if model_file:
                    saver.restore(sess, model_file)
                    print("Load from latest checkpoint")
            else:
                os.mkdir(opt.model_dir)

            for epoch in range(opt.max_epoch):

                _ = sess.run(global_step)

                # Get all images and labels
                index_epoch = sess.run(alldata.index_dequeue_op)
                train_data_epoch = np.array(alldata.train_data)[
                    index_epoch]
                train_labels_epoch = np.array(alldata.train_labels)[
                    index_epoch]

                sess.run(alldata.iterator.initializer, feed_dict={
                         alldata.image_paths_placeholder: train_data_epoch,
                         alldata.labels_placeholder: train_labels_epoch})

                # Training
                num_batch = 0
                while num_batch < alldata.num_batches:

                    tensor_list = [train_op, logits, center_loss,
                                   cross_entropy_mean, total_loss, accuracy, global_step]
                    feed_dict = {alldata.phase_train_placeholder: True,
                                 alldata.batch_size_placeholder: opt.batch_size,
                                 alldata.keep_probability_placeholder: opt.keep_prob}

                    _, logits_, center_loss_, cross_entropy_mean_, total_loss_, accuracy_, global_step_ = sess.run(
                        tensor_list, feed_dict=feed_dict)
                    
                    if num_batch % 5 == 0:

                        # Visualization
                        summary_str = sess.run(
                            summary_op, feed_dict=feed_dict)
                        
                        train_writer.add_summary(
                            summary_str, global_step=global_step_)
                        
                        print('epoch:%d/%d' % (epoch, opt.max_epoch))
                        print("Step: %d/%d, accuracy: %3f, center loss: %4f, cross loss: %4f, Total Loss: %4f" % (
                            global_step_, alldata.num_batches*opt.max_epoch, accuracy_, center_loss_, cross_entropy_mean_, total_loss_))

                        # Saver
                        saver.save(sess, opt.model_dir+'model.ckpt',
                                   global_step=global_step_)
                    num_batch += 1

                train_writer.add_summary(summary_str, global_step=global_step_)

                # Validation
                val_data_epoch = np.array(alldata.val_data)[
                    :opt.batch_size*alldata.num_val_batches]
                val_labels_epoch = np.array(alldata.val_labels)[
                    :opt.batch_size*alldata.num_val_batches]

                sess.run(alldata.iterator.initializer, feed_dict={
                         alldata.image_paths_placeholder: val_data_epoch,
                         alldata.labels_placeholder: val_labels_epoch})
                
                loss_val_mean, center_loss_val_mean, cross_entropy_mean_val_mean, accuracy_val_mean = 0,0,0,0

                for num_batch in range(alldata.num_val_batches):
                    
                    tensor_list = [total_loss, center_loss,
                                   cross_entropy_mean, accuracy, summary_op]
                    feed_dict = {alldata.phase_train_placeholder: False,
                                 alldata.keep_probability_placeholder: 1.0}

                    loss_val, center_loss_val, cross_entropy_mean_val, accuracy_val, summary_val = sess.run(
                        tensor_list, feed_dict=feed_dict)
                    
                    loss_val_mean += loss_val
                    center_loss_val_mean += center_loss_val
                    cross_entropy_mean_val_mean += cross_entropy_mean_val
                    accuracy_val_mean += accuracy_val

                val_writer.add_summary(summary_val, global_step=epoch)
                loss_val_mean/=alldata.num_val_batches
                center_loss_val_mean/=alldata.num_val_batches
                cross_entropy_mean_val_mean/=alldata.num_val_batches
                accuracy_val_mean/=alldata.num_val_batches
                print("Validation Result: accuracy: %3f, center loss: %4f, cross entropy loss: %4f, Total loss: %4f" % (accuracy_val_mean, center_loss_val_mean, cross_entropy_mean_val_mean,loss_val_mean))
            
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    train()


                    
                    
