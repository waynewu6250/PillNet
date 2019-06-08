import numpy as np
import tensorflow as tf
from model import InceptionResnetV1


class PillNet:
    def __init__(self, image_batch, label_batch, num_classes, opt):
        self.inference_model = InceptionResnetV1()
        self.image_batch = image_batch
        self.label_batch = label_batch
        self.num_classes = num_classes
        self.opt = opt

    def layer_batch_norm(self, x, n_out, phase_train):
        beta = tf.get_variable("beta",
                               [n_out],
                               initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        gamma = tf.get_variable("gamma",
                                [n_out],
                                initializer=tf.constant_initializer(value=1.0, dtype=tf.float32))

        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema_mean, ema_var))

        reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
        normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
                                                            beta, gamma, 1e-3, True)
        return tf.reshape(normed, [-1, n_out])

    def linear(self, input, weight_shape, bias_shape, phase_train):

        W = tf.get_variable("W",
                            weight_shape,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b",
                            bias_shape,
                            initializer=tf.constant_initializer(0))
        logits = tf.matmul(input, W)+b
        return tf.nn.relu(self.layer_batch_norm(logits, weight_shape[1], phase_train))

    def center_loss_op(self, logits, labels, alpha):
        
        feature_size = logits.get_shape()[1]
        centers = tf.get_variable('centers', [self.num_classes, feature_size],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
        
        # Get each batch's center features
        centers_batch = tf.gather(centers, tf.reshape(labels,[-1]))

        # Update the centers first
        # Calculate the difference between features and its label centers_batches
        diff = alpha*(centers_batch-logits)
        centers = tf.scatter_sub(centers, labels, diff)
        
        # Then calculate center loss
        with tf.control_dependencies([centers]):
            center_loss = tf.reduce_mean(tf.square(centers_batch-logits))
        
        return center_loss, centers


    def inference(self, keep_prob, phase_train, embedding_size, weight_decay):

        # Logits Inference
        logits, _ = self.inference_model.inference(self.image_batch,
                                                    keep_prob, phase_train, embedding_size, weight_decay)
        # Linear layer
        with tf.variable_scope("fc"):
            outputs = self.linear(logits, [logits.get_shape()[1], self.num_classes],
                                [self.num_classes], phase_train)

        # Center Loss
        center_loss, _ = self.center_loss_op(
            logits, self.image_batch, self.opt.alpha)
        tf.identity(center_loss, name='center_loss')
        tf.summary.scalar('center_loss', center_loss)

        # Softmax Cross Entropy Loss
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, self.label_batch)
        cross_entropy_mean = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')
        tf.identity(cross_entropy_mean, name='cross_entropy_mean')
        tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)

        # Total Loss
        L2_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss=cross_entropy_mean+self.opt.center_loss_factor*center_loss+L2_loss
        tf.identity(total_loss, name='total_loss')
        tf.summary.scalar('total_loss', total_loss)
        
        # Accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(self.label_batch, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.identity(accuracy, name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        return total_loss, accuracy

    def optimize(self, total_loss, update_grads, global_step):
        
        optimizer = tf.train.AdamOptimizer(self.opt.lr)
        grads = optimizer.compute_gradients(total_loss, update_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Visualization
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
        
        variable_averages=tf.train.ExponentialMovingAverage(self.opt.moving_average_decay,global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        
        return train_op
        


