
import argparse
from model import *
import tensorflow as tf
from glob import glob
import os
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--test_dir', dest='test_dir', default='./test/text', help='test sample are saved here')
parser.add_argument('--test_set', dest='test_set', default='moire_rgb_text', help='dataset of base layer for testing')
args = parser.parse_args()


class Test():
    def __init__(self, sess, input_c_dim=3):
        self.sess = sess
        self.input_c_dim = input_c_dim
        # build model
        self.X = tf.placeholder(tf.float32, [None, 512, 512, self.input_c_dim],
                                 name='moire_image')     
        self.patch_size = 512

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.Y = generator(self.X, self.patch_size, self.patch_size, 1, is_training=self.is_training)
        print("[*] Initialize model successfully...")

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            #full_path = tf.train.latest_checkpoint(checkpoint_dir)
            full_path = './checkpoint/Demoire-tensorflow-69581'
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0


    def main(self, test_files, ckpt_dir, save_dir):
        """Test"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        print("[*] " + " start testing...")

        for idx in range(len(test_files)):
            moire_image = load_images(test_files[idx]).astype(np.float32) / 255.0 

            demoire_image = self.sess.run(self.Y, feed_dict={self.X: moire_image, self.is_training: False})        
            print(test_files[idx])       
            num_img = test_files[idx][43:-6]  
            print("num_img=",num_img)   

            moireimage = np.clip(255 * moire_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * demoire_image, 0, 255).astype('uint8')

            #save_images(os.path.join(save_dir, num_img+'_m.png'), moireimage)
            save_images(os.path.join(save_dir, num_img+'_dm.png'), outputimage)

if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = Test(sess)
            test_files = glob('../RAW_img_dm/data/testset3/{}/*.png'.format(args.test_set))
            model.main(test_files, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = Test(sess)
            test_files_base = glob('../RAW_img_dm/data/testset3/{}/*.png'.format(args.test_set_base))
            model.main(test_files_base, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)