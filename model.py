import time
import numpy as np
import tensorflow.contrib.slim as slim
from utils import *
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import scipy.io as sio
import utils
from scipy.misc import imread
from PIL import Image
import vgg

def aspp(input, input_channels, reuse=False):
    with tf.variable_scope('Aspp') as scope:
        if reuse:
            scope.reuse_variables()
        p1 = tf.layers.conv2d(input, input_channels, 1, padding='same', name='aspp_p1')
        p1 = tf.nn.relu(p1)
        p2 = tf.layers.conv2d(input, input_channels, 3, padding='same', dilation_rate=(6, 6), name='aspp_p2')
        p2 = tf.nn.relu(p2)
        p3 = tf.layers.conv2d(input, input_channels, 3, padding='same', dilation_rate=(12, 12), name='aspp_p3')
        p3 = tf.nn.relu(p3)
        p4 = tf.layers.conv2d(input, input_channels, 3, padding='same', dilation_rate=(18, 18), name='aspp_p4')
        p4 = tf.nn.relu(p4)
        p5 = tf.nn.avg_pool(input, ksize=[1, input.shape[1], input.shape[2], 1], strides=[1, 1, 1, 1], padding='VALID', name='aspp_pool')
        p5 = tf.image.resize_images(p5, size=(input.shape[1], input.shape[2]), method=0)
        output = tf.concat([p1, p2, p3, p4, p5], 3)
        output = tf.layers.conv2d(output, input_channels, 1, padding='same', name='aspp_bottle')
        output = tf.nn.relu(output)
    return output

def generator(input, patch_size1, patch_size2, batch_size=8, is_training=True, reuse=False, output_channels=3):
    with tf.variable_scope('Generator') as scope:
        if reuse:
            scope.reuse_variables()
        with tf.variable_scope('conv_in'):
            conv_in = tf.layers.conv2d(input, 32, 7, padding='same', name='conv1_1')
            conv_in = tf.nn.relu(conv_in) 
        # bias
        # encoder
        with tf.variable_scope('conv1_1'):
            output = tf.layers.conv2d(conv_in, 32, 3, padding='same', name='conv1_1')
            output = tf.nn.relu(output) 
        with tf.variable_scope('conv1_2'):
            output = tf.layers.conv2d(output, 32, 3, padding='same', name='conv1_2')
            conv1_2 = tf.nn.relu(output)     
        with tf.variable_scope('conv2_1'):
            output = tf.layers.conv2d(conv1_2, 64, 3, strides=(2, 2), padding='same', name='conv2_1')
            output = tf.nn.relu(output)  
        with tf.variable_scope('conv2_2'):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv2_2')
            conv2_2 = tf.nn.relu(output)  
        with tf.variable_scope('conv3_1'):
            output = tf.layers.conv2d(conv2_2, 128, 3, strides=(2, 2), padding='same', name='conv3_1')
            output = tf.nn.relu(output)  
        with tf.variable_scope('conv3_2'):
            output = tf.layers.conv2d(output, 128, 3, padding='same', name='conv3_2')
            conv3_2 = tf.nn.relu(output) 
            conv3_2 = aspp(conv3_2, 128, reuse)
              
        # decoder
        with tf.variable_scope('deconv3_1'):
            output = tf.layers.conv2d(conv3_2, 128, 3, padding='same', name='deconv3_1')
            output = tf.nn.relu(output)  
        with tf.variable_scope('deconv2_2'):
            output = tf.image.resize_images(output, size=(patch_size1//2, patch_size2//2), method=1)  
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='deconv2_2')
            output = tf.nn.relu(output) 
            #output_b_aspp = output
            output += aspp(conv2_2, 64, reuse)
            #output_a_aspp = output
        with tf.variable_scope('deconv2_1'):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='deconv2_1')
            output = tf.nn.relu(output)  
        with tf.variable_scope('deconv1_2'):
            output = tf.image.resize_images(output, size=(patch_size1, patch_size2), method=1)  
            output = tf.layers.conv2d(output, 32, 3, padding='same', name='deconv1_2')
            output = tf.nn.relu(output) 
           # output_b_aspp = output
            output += aspp(conv1_2, 32, reuse)
            #output_a_aspp = output
        with tf.variable_scope('deconv1_1'):
            output = tf.layers.conv2d(output, 32, 3, padding='same', name='deconv1_1')
            output = tf.nn.relu(output)  
        with tf.variable_scope('deconv1_0'):
            output = tf.layers.conv2d(output, output_channels, 3, padding='same', name='deconv1_0')
            output += input
            output_mid = output


        # multiply
        r = 16
        pool_size = [1, input.shape[1], input.shape[2], 1]
        with tf.variable_scope('multi_1'):
            multi_1 = tf.layers.conv2d(output, 64, 3, padding='same', name='multi_1')
            se = tf.nn.avg_pool(multi_1, pool_size, [1, 1, 1, 1], padding='VALID')
            se = tf.layers.conv2d(se, 64/r, 1, padding='same', use_bias=True)
            se = tf.nn.relu(se)
            se = tf.layers.conv2d(se, 64, 1, padding='same', use_bias=True) 
            se = tf.sigmoid(se)
            multi_1 = se * multi_1
            multi_1 = tf.nn.relu(multi_1) 
        with tf.variable_scope('multi_2'):
            multi_2 = tf.layers.conv2d(multi_1, 64, 3, padding='same', name='multi_2')
            se = tf.nn.avg_pool(multi_2, pool_size, [1, 1, 1, 1], padding='VALID')
            se = tf.layers.conv2d(se, 64/r, 1, padding='same', use_bias=True)
            se = tf.nn.relu(se)
            se = tf.layers.conv2d(se, 64, 1, padding='same', use_bias=True) 
            se = tf.sigmoid(se)
            multi_2 = se * multi_2
            multi_2 = tf.nn.relu(multi_2) 
        with tf.variable_scope('multi_3'):
            multi_3 = tf.layers.conv2d(multi_2, 64, 3, padding='same', name='multi_3')
            se = tf.nn.avg_pool(multi_3, pool_size, [1, 1, 1, 1], padding='VALID')
            se = tf.layers.conv2d(se, 64/r, 1, padding='same', use_bias=True)
            se = tf.nn.relu(se)
            se = tf.layers.conv2d(se, 64, 1, padding='same', use_bias=True) 
            se = tf.sigmoid(se)
            multi_3 = se * multi_3
            multi_3 = tf.nn.relu(multi_3) 
        with tf.variable_scope('multi_4'):
            multi_4 = tf.layers.conv2d(multi_3, 64, 3, padding='same', name='multi_4')
            se = tf.nn.avg_pool(multi_4, pool_size, [1, 1, 1, 1], padding='VALID')
            se = tf.layers.conv2d(se, 64/r, 1, padding='same', use_bias=True)
            se = tf.nn.relu(se)
            se = tf.layers.conv2d(se, 64, 1, padding='same', use_bias=True) 
            se = tf.sigmoid(se)
            multi_4 = se * multi_4
            multi_4 = tf.nn.relu(multi_4)     
        with tf.variable_scope('multi_5'):
            multi_5 = tf.layers.conv2d(multi_4, output_channels, 3, padding='same', name='multi_5')
        tf.add_to_collection('conv_output', multi_5)    
        tf.add_to_collection('conv_output', output)  
        multi_out = tf.nn.relu(output * multi_5)
        tf.add_to_collection('conv_output', multi_out) 

        return multi_out


def discriminator(input, is_training=True, reuse=False):
    with tf.variable_scope('Discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        size = 64
        res = [input]
        d = tcl.conv2d(input, num_outputs=size, kernel_size=4, stride=2, padding='SAME',
                       weights_initializer=tf.random_normal_initializer(0, 0.02))
        d = tf.nn.leaky_relu(d, 0.2)
        res.append(d)

        d = tcl.conv2d(d, num_outputs=size * 2, kernel_size=4, stride=2, padding='SAME',
                       weights_initializer=tf.random_normal_initializer(0, 0.02))
        # d = tcl.instance_norm(d)
        d = tf.nn.leaky_relu(d, 0.2)
        res.append(d)

        d = tcl.conv2d(d, num_outputs=size * 4, kernel_size=4, stride=2, padding='SAME',
                       weights_initializer=tf.random_normal_initializer(0, 0.02))
        # d = tcl.instance_norm(d)
        d = tf.nn.leaky_relu(d, 0.2)
        res.append(d)

        d = tcl.conv2d(d, num_outputs=size * 8, kernel_size=4, stride=1, padding='SAME',
                weights_initializer=tf.random_normal_initializer(0, 0.02))
        # d = tcl.instance_norm(d)
        d = tf.nn.leaky_relu(d, 0.2)
        res.append(d)

        d = tcl.conv2d(d, num_outputs=1, kernel_size=4, stride=1, padding='SAME',
                weights_initializer=tf.random_normal_initializer(0, 0.02))
        res.append(d)
        return res[1:]

def criterionGAN(d_images, batch_size, target_bool):
    loss = 0
    d_images = d_images[-1]
    # print(d_images)
    # print(batch_size)
    if target_bool:
        for i in range(batch_size):
            d_image = d_images[i]
            # print(d_image.shape)
            loss += tf.nn.l2_loss(d_image - 1.0)
        return loss / batch_size
    else:
        for i in range(batch_size):
            d_image = d_images[i]
            loss += tf.nn.l2_loss(d_image)
        return loss / batch_size   


class demoire(object):
    def __init__(self, sess, data, args, input_c_dim=3):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.patch_size = 128
        self.data = data
        self.args = args

        # build model
        self.X = tf.placeholder(tf.float32, [args.batch_size, self.patch_size, self.patch_size, self.input_c_dim],
                                 name='moire_image')
        self.Y_ = tf.placeholder(tf.float32, [args.batch_size, self.patch_size, self.patch_size, self.input_c_dim],
                                 name='clean_image')                     
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.Y = generator(self.X, self.patch_size, self.patch_size, args.batch_size, is_training=self.is_training)

        self.D_real = discriminator(self.Y_)
        self.D_fake = discriminator(self.Y, reuse=True)

        # calculate loss
        self.loss = (1.0 / args.batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)  # MSE loss
        # define perceptual loss
        CONTENT_LAYER = 'relu5_4'
        vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
        demoire_vgg = vgg.net(vgg_dir, vgg.preprocess(self.Y * 255))
        clean_vgg = vgg.net(vgg_dir, vgg.preprocess(self.Y_ * 255))
        content_size = 128*128*3*args.batch_size/16/16
        self.loss_content = 2 * tf.nn.l2_loss(demoire_vgg[CONTENT_LAYER] - clean_vgg[CONTENT_LAYER]) / content_size  
        self.loss_sum = self.loss + self.loss_content
        # define gan loss
        self.loss_d_fake = criterionGAN(self.D_fake, args.batch_size, False)
        self.loss_d_real = criterionGAN(self.D_real, args.batch_size, True)
        self.G_loss = criterionGAN(self.D_fake, args.batch_size, True)
        self.D_loss = (self.loss_d_fake + self.loss_d_real) * 0.5
        # GAN feature matching loss
        self.loss_G_GAN_Feat = 0
        for i in range(len(self.D_fake)-1):
            self.loss_G_GAN_Feat += tf.reduce_mean(abs(self.D_real[i]-self.D_fake[i])) / 4.0
        self.G_loss_sum = self.G_loss + self.loss_sum + self.loss_G_GAN_Feat * 1000

        self.G_vars = [var for var in tf.trainable_variables() if var.name.startswith('Generator')]
        self.D_vars = [var for var in tf.trainable_variables() if var.name.startswith('Discriminator')]

        self.lr = args.lr
        self.fig_count = 30
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)    
        max_steps = args.epoch * int(self.data.data_shape() // args.batch_size)
        self.lr = tf.train.polynomial_decay(self.lr, self.global_step, max_steps, end_learning_rate=0.0,
                                    power=0.2)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_sum, self.global_step, self.G_vars)
            self.train_op_g = tf.train.AdamOptimizer(self.lr/50.).minimize(self.G_loss_sum, var_list=self.G_vars)
            self.train_op_d = tf.train.AdamOptimizer(self.lr/50.).minimize(self.D_loss, var_list=self.D_vars)   

        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def train(self, batch_size, ckpt_dir, epoch, sample_dir, eval_every_epoch=1):
        # assert data range is between 0 and 1
        dataShape = self.data.data_shape()
        numBatch = int(dataShape // batch_size)
                       
        # load pretrained model
        load_model_status, iter_num = self.load(ckpt_dir)
        if load_model_status:
            start_epoch = iter_num // numBatch
            start_step = 0
            print("[*] Model restore success!")
            print("start epoch = %d, start iter_num = %d" % (start_epoch, iter_num))
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('loss_content', self.loss_content)
        tf.summary.scalar('loss_sum', self.loss_sum)
        tf.summary.scalar('G_loss', self.G_loss)
        tf.summary.scalar('D_loss', self.D_loss)
        tf.summary.scalar('G_loss_sum', self.G_loss_sum)
        tf.summary.scalar('loss_G_GAN_Feat', self.loss_G_GAN_Feat)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()

        for epoch in range(start_epoch, epoch):   
            order = np.arange(0, numBatch, 1)
            random.shuffle(order)  
            for batch_id in range(start_step, numBatch):
                [data_in, data_out] = self.data(batch_size)
                batch_in = data_in[order[batch_id]*batch_size:(order[batch_id]+1)*batch_size,:,:,:]
                batch_out = data_out[order[batch_id]*batch_size:(order[batch_id]+1)*batch_size,:,:,:]

                batch_in = batch_in / 255.0
                batch_out = batch_out / 255.0

                _, _, _, loss_content, loss, loss_sum, G_loss, D_loss, loss_G_GAN_Feat, G_loss_sum, summary \
                    = self.sess.run([self.train_op, self.train_op_g, self.train_op_d, self.loss_content, self.loss, self.loss_sum, self.G_loss, self.D_loss, self.loss_G_GAN_Feat, self.G_loss_sum, merged], 
                                    feed_dict={self.X: batch_in, self.Y_: batch_out, self.is_training: True})

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f loss: %.6f loss_content: %.6f loss_sum: %.6f G_loss: %.6f D_loss: %.6f loss_G_GAN_Feat:  %.6f G_loss_sum: %.6f"
                        % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss, loss_content, loss_sum, G_loss, D_loss, loss_G_GAN_Feat, G_loss_sum))                               
                
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                # evaluate added
                eval_data_in = batch_in[:16, :, :, :]
                eval_data_out = batch_out[:16, :, :, :]
                samples = self.sess.run(self.Y, feed_dict={self.X: eval_data_in, self.is_training: False})
                eval_data_out = np.clip(255 * eval_data_out, 0, 255).astype('uint8')
                samples = np.clip(255 * samples, 0, 255).astype('uint8')                
                fig = data2fig(eval_data_out)
                plt.savefig('{}/{}_gt.png'.format(sample_dir, str(self.fig_count).zfill(3)), bbox_inches='tight')    
                plt.close(fig) 
                
                eval_data_in = np.clip(255 * eval_data_in, 0, 255).astype('uint8')
                fig = data2fig(eval_data_in)
                plt.savefig('{}/{}_m.png'.format(sample_dir, str(self.fig_count).zfill(3)), bbox_inches='tight')    
                plt.close(fig) 
                
                fig = data2fig(samples)
                plt.savefig('{}/{}_dm.png'.format(sample_dir, str(self.fig_count).zfill(3)), bbox_inches='tight')        
                plt.close(fig)    
                self.fig_count += 1    
            self.save(iter_num, ckpt_dir)
            self.global_step = iter_num
        print("[*] Finish training.")

    def save(self, iter_num, ckpt_dir, model_name='Demoire-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0
