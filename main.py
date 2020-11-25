import argparse
from glob import glob
import tensorflow as tf
from model import demoire
from utils import *
import h5py
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=20, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0005, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test or val or plot_feature_map')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--file_in', dest='file_in', default='./trainset/datain.h5', help='datain are saved here')
parser.add_argument('--file_out', dest='file_out', default='./trainset/dataout.h5', help='dataout are saved here')
#parser.add_argument('--val_in', dest='val_in', default='../../data3/valin.h5', help='valin are saved here')
#parser.add_argument('--val_out', dest='val_out', default='../../data3/valout.h5', help='valout are saved here')
args = parser.parse_args()
tf.reset_default_graph() 

class Sampler(object):
    def __init__(self):
        self.name = "demoire"
        [self.cur_batch_in, self.cur_batch_out] = self.load_new_data()
        self.train_batch_idx = 0

    def load_new_data(self):
        moire_rgb_pt = h5py.File(args.file_in, 'r')
        gt_rgb_pt = h5py.File(args.file_out, 'r')
        for key1 in moire_rgb_pt.keys():
            data_in = moire_rgb_pt[(key1)]
        for key2 in gt_rgb_pt.keys():
            data_out = gt_rgb_pt[(key2)]
        return data_in, data_out

    def __call__(self, batch_size=args.batch_size):        
        return self.cur_batch_in, self.cur_batch_out

    def data_shape(self):
        return self.cur_batch_in.shape[0]

data = Sampler()

def demoire_train(demoire):   
    demoire.train(batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, sample_dir=args.sample_dir)

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)   
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    gpu_options.allow_growth = True
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = demoire(sess, data, args)
        demoire_train(model)

if __name__ == '__main__':
    tf.app.run()
