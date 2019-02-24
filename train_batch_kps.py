import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import time
import datetime
from  data import ZebrishData
import config as cfg 
from timer import Timer


slim = tf.contrib.slim
batch_size = cfg.batch_size

def decoder_arg_scope(weight_decay = 0.1, stddev = 0.01, is_training = True, batch_norm_var_collection = 'moving_vars'):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training,
        'variables_collections':{
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection]
        }
    }
    with slim.arg_scope([slim.conv2d], weights_regularizer = slim.l2_regularizer(weight_decay),
                        weights_initializer = tf.truncated_normal_initializer(stddev = stddev),
                        activation_fn = tf.nn.relu,
                        normalizer_fn = slim.batch_norm,
                        normalizer_params = batch_norm_params) as sc:
        return sc

def decoder_base(inputs, is_training = True, scope = None):
    with tf.variable_scope(scope, 'Decoder', [inputs]):
        with slim.arg_scope(decoder_arg_scope(is_training = is_training)):
            with tf.variable_scope('Conv1'):
                net = slim.conv2d(inputs, 256, [3, 3], scope = 'Conv2d_0a_3x3')
                net = slim.conv2d(inputs, 256, [3, 3], scope = 'Conv2d_0b_3x3')
                net = slim.conv2d(inputs, 256, [3, 3], scope = 'Conv2d_0c_3x3')
            
            with tf.variable_scope('Deconv1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256]), name = 'weights')
                biases = tf.Variable(tf.constant(0.1, shape = [128]), name= 'biases')
                net = tf.nn.conv2d_transpose(net, kernel, 
                                            tf.stack([tf.shape(net)[0], 14, 14, 128]), 
                                            strides = [1, 2, 2, 1], padding = 'SAME')
                net = tf.nn.bias_add(net, biases)
                net = tf.nn.relu(net, name = "relu")

            with tf.variable_scope('Conv2'):
                net = slim.conv2d(net, 128, [3, 3], scope = 'Conv2d_0a_3x3')
                net = slim.conv2d(net, 128, [3, 3], scope = 'Conv2d_0b_3x3')
                net = slim.conv2d(net, 128, [3, 3], scope = 'Conv2d_0c_3x3')

            with tf.variable_scope('Deconv2'):
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128]), name= 'weights')
                biases = tf.Variable(tf.constant(0.1, shape = [64]), name= 'biases')
                net = tf.nn.conv2d_transpose(net, kernel, 
                                            tf.stack([tf.shape(net)[0], 28, 28, 64]),
                                            strides = [1, 2, 2, 1], padding= 'SAME')
                net = tf.nn.bias_add(net, biases)
                net = tf.nn.relu(net, name = "relu")
            
            with tf.variable_scope('Conv3'):
                net = slim.conv2d(net, 64, [3, 3], scope = 'Conv2d_0a_3x3')
                net = slim.conv2d(net, 64, [3, 3], scope = 'Conv2d_0b_3x3')

            with tf.variable_scope('Deconv3'):
                kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 64]), name= 'weights')
                biases = tf.Variable(tf.constant(0.1, shape = [32]), name= 'biases')
                net = tf.nn.conv2d_transpose(net, kernel, 
                                            tf.stack([tf.shape(net)[0], 56, 56, 32]),
                                            strides = [1, 2, 2, 1], padding= 'SAME')
                net = tf.nn.bias_add(net, biases)
                net = tf.nn.relu(net, name = "relu")
            
            with tf.variable_scope('Conv4'):
                net = slim.conv2d(net, 32, [3, 3], scope = 'Conv2d_0a_3x3')
                net = slim.conv2d(net, 32, [3, 3], scope = 'Conv2d_0b_3x3')

            with tf.variable_scope('Conv6'):
                net = slim.conv2d(net, 21, [3, 3], scope = 'Conv2d_0a_3x3')
            
            return net


def encoder(inputs, is_training = True, scope = None):
    with tf.variable_scope(scope, 'Encoder', [inputs]):
        with slim.arg_scope(decoder_arg_scope(is_training = is_training)):
            # block1
            net = slim.conv2d(inputs, 32, [3, 3], scope = "conv1_1")
            net = slim.conv2d(net, 32, [3, 3], scope = 'conv1_2')
            net = slim.max_pool2d(net, [2, 2], stride = 2, padding = "SAME", scope = "pool1")

            # block2            
            net = slim.conv2d(net, 64, [3, 3], scope = "conv2_1")          
            net = slim.conv2d(net, 64, [3, 3], scope = "conv2_2")
            net = slim.max_pool2d(net, [2, 2], stride = 2, padding = "SAME", scope = "pool2")

            # block3            
            net = slim.conv2d(net, 128, [3, 3], scope = "conv3_1")            
            net = slim.conv2d(net, 128, [3, 3], scope = "conv3_2")
            net = slim.conv2d(net, 128, [3, 3], scope = "conv3_3")
            net = slim.max_pool2d(net, [2, 2], stride = 2, padding = "SAME", scope = "pool3")

            # block4
            net = slim.conv2d(net, 256, [3, 3], scope = "conv4_1")
            net = slim.conv2d(net, 256, [3, 3], scope = "conv4_2")
            net = slim.conv2d(net, 256, [3, 3], scope = "conv4_3")

            return net

def loss_fn(logits, labels, labels_weights):
    # logits' shape = [batch_size, 56, 56, 21]
    # labels' shape = [batch_size, 21]

    print(logits.shape)
    logits = tf.transpose(logits, [0, 3, 1, 2])
    print(logits.shape, labels.shape)
    # logits_shape = tf.shape(logits)
    logits_shape = logits.get_shape()
    logits = tf.reshape(logits, [-1, logits_shape[1], logits_shape[2] * logits_shape[3]])
    

    labels_one_hot = tf.one_hot(labels, 56*56)
    # print(logits.shape, labels_one_hot.shape)
    loss = tf.losses.softmax_cross_entropy(labels_one_hot, logits)
    # print(loss.shape)
    # loss = tf.multiply(loss, labels_weights)
    # loss /= tf.to_float(tf.shape(logits)[0])
    loss = loss * cfg.batch_size * cfg.keypoints_num
    loss_keypoints = tf.keras.backend.switch(tf.reduce_sum(labels_weights) > 0,
                        lambda:  loss / tf.reduce_sum(labels_weights),
                        lambda: tf.constant(0.0)
                        )

    tf.losses.add_loss(loss_keypoints)
    return loss_keypoints 

def cover_loss(logits, labels_weights):
    logits_shape = logits.get_shape() 
    logits = tf.reshape(logits, (-1, logits_shape[2] * logits_shape[3]))
    print(logits.shape)
    print(labels_weights.shape)
    negetive_index = tf.where(labels_weights>0)[:, 0]

    negetive_logits = tf.gather(logits, negetive_index)
    print(negetive_logits)
    negetive_prob = tf.nn.softmax(negetive_logits)
    negetive_max = tf.reduce_max(negetive_prob, axis=1)
    negetive_max = 1 - negetive_max

    loss = -tf.reduce_sum(tf.log(negetive_max)) * 100 / tf.reduce_sum(labels_weights)
    return loss
        
    
def cover_loss_cond(logits, labels_weights):
    
    logits = tf.transpose(logits, [0, 3, 1, 2])
    labels_weights = 1 - tf.reshape(labels_weights, (-1,1))
    loss = tf.keras.backend.switch(tf.reduce_sum(labels_weights) > 0,  
                    lambda: cover_loss(logits, labels_weights),
                    lambda: tf.constant(0.0)
                    )
    tf.losses.add_loss(loss)
    return loss

def run_training():
    # cuda 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # input data 
    imgs_ph = tf.placeholder(tf.float32, shape = [batch_size, 56, 56, 1])
    labels_ph = tf.placeholder(tf.int32, shape = [batch_size, 21])
    labels_weights_ph = tf.placeholder(tf.float32, shape = [batch_size, 21]) 
    is_training = tf.placeholder(tf.bool)
    zebrishdata = ZebrishData('train')

    # network and loss
    net = encoder(imgs_ph, is_training = is_training)
    logits = decoder_base(net, is_training = is_training)
    loss_keypoints = loss_fn(logits, labels_ph, labels_weights_ph)
    loss_cover = cover_loss_cond(logits, labels_weights_ph)
    tf.summary.scalar('cover_loss', loss_cover)
    tf.summary.scalar('keypoint_loss', loss_keypoints)
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('total_loss', total_loss)

    # summary operation
    output_dir = ('output_with_crop')
    ckpt_dir = os.path.join(output_dir, 'keypoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(ckpt_dir, flush_secs=60)


    # optimize operation
    global_step = tf.get_variable(
        'global_step', [], tf.int64,
        initializer=tf.zeros_initializer(), trainable=False)
    # sess.run(global_step_tensor.initializer)
    # global_step = tf.train.create_global_step()
    # global_step = tf.train.global_step(sess, global_step_tensor)
    learning_rate = tf.train.exponential_decay(
        cfg.learning_rate, global_step, cfg.decay_steps,
        cfg.decay_rate, True, name='learning_rate')
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = slim.learning.create_train_op(
        total_loss, optimizer, global_step=global_step
    )
    
     # session and saver
    saver = tf.train.Saver()
    tfconfig = tf.ConfigProto()
    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 1.0        
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'output_with_crop/keypoints-11000')

    writer.add_graph(sess.graph)
    
    train_timer = Timer()
    global_step_val = global_step.eval(sess)
    for step in range(global_step_val+1, cfg.max_iter+1):  
        load_timer = Timer()
        load_timer.tic()
        imgs, labels, labels_weights = zebrishdata.get()
        load_timer.toc()            
        
        feed_dict = {imgs_ph:imgs , labels_ph:labels, labels_weights_ph:labels_weights, is_training:True}
        if step % cfg.summary_iter == 0:
                if step % (cfg.summary_iter * 10) == 0:

                    train_timer.tic()
                    summary_str, total_loss_val, loss_val, cover_loss_val, _ = sess.run(
                        [summary_op, total_loss, loss_keypoints, loss_cover, train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = '''{} Epoch: {}, Step: {}, Learning rate: {:.5f}, Total_loss: {:5.3f}, keypoint_loss: {:5.3f}, cover_loss is: {:5.3f}\nSpeed: {:.3f}s/iter,Load: {:.3f}s/iter, Remain: {}'''.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        zebrishdata.epoch,
                        int(step),
                        round(learning_rate.eval(session=sess), 6),
                        total_loss_val,
                        loss_val,
                        cover_loss_val,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, cfg.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = sess.run(
                        [summary_op, train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                writer.add_summary(summary_str, step)

        else:
            train_timer.tic()
            sess.run(train_op, feed_dict=feed_dict)
            train_timer.toc()

        if step % cfg.save_iter == 0:
            print('{} Saving checkpoint file to: {}'.format(
                datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                output_dir))
            saver.save(
                sess, ckpt_dir, global_step=global_step)
        if step == cfg.max_iter:
            print('{} Saving checkpoint file to: {}'.format(
                datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                output_dir))
            saver.save(
                sess, ckpt_dir, global_step=global_step)
        
      

if __name__ == "__main__":
    run_training()
    # inference()
    