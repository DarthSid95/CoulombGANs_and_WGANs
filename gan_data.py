from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import glob
from absl import flags
import csv

from scipy import io as sio

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import PdfPages


### Need to prevent tfds downloads bugging out? check
import urllib3
urllib3.disable_warnings()


FLAGS = flags.FLAGS

'''
GAN_DATA functions are specific to the topic, ELeGANt, RumiGAN, PRDeep or DCS. Data reading and dataset making functions per data, with init having some specifics generic to all, such as printing instructions, noise params. etc.
'''
'''***********************************************************************************
********** GAN_DATA_Baseline *********************************************************
***********************************************************************************'''
class GAN_DATA_Base():

	def __init__(self):#,data,testcase,number,out_size):
		# self.gen_func = 'self.gen_func_'+data+'()'
		# self.dataset_func = 'self.dataset_'+data+'(self.train_data)'
		self.noise_mean = 0.0
		self.noise_stddev = 1.00

		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data)'
		self.noise_mean = 0.0
		self.noise_stddev = 1.00
		#Default Number of repetitions of a dataset in tf.dataset mapping
		self.reps = 1
		if self.loss == 'RBF':
			self.reps_centres = int(np.ceil(self.N_centers//self.batch_size))
		if self.data == 'g2':
			self.MIN = -1
			self.MAX = 1.2
			self.noise_dims = 100
			self.output_size = 2
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		elif self.data == 'gmm8':
			self.noise_dims = 100
			self.output_size = 2
			self.noise_mean = 0.0
			self.noise_stddev = 1.0


	def gen_func_g2(self):

		self.MIN = -5.5
		self.MAX = 10.5
		self.data_centres = tf.random.normal([500*self.N_centers, 2], mean =np.array([3.5,3.5]), stddev = np.array([1.25,1.25]))
		data = tf.random.normal([500*self.batch_size.numpy(), 2], mean =np.array([3.5,3.5]), stddev = np.array([1.25,1.25]))

	def dataset_g2(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)

		if self.loss == 'RBF':
			self.center_dataset = tf.data.Dataset.from_tensor_slices((self.data_centres))
			self.center_dataset = self.center_dataset.shuffle(4)
			self.center_dataset = self.center_dataset.batch(self.N_centers)
			self.center_dataset = self.center_dataset.prefetch(5)

		return train_dataset


	def gen_func_gmm8(self):
		## Cirlce - [0,1]
		scaled_circ = 0.35
		offset = 0.5
		locs = [[scaled_circ*1.+offset, 0.+offset], \
				[0.+offset, scaled_circ*1.+offset], \
				[scaled_circ*-1.+offset,0.+offset], \
				[0.+offset,scaled_circ*-1.+offset], \
				[scaled_circ*1*0.7071+offset, scaled_circ*1*0.7071+offset], \
				[scaled_circ*-1*0.7071+offset, scaled_circ*1*0.7071+offset], \
				[scaled_circ*1*0.7071+offset, scaled_circ*-1*0.7071+offset], \
				[scaled_circ*-1*0.7071+offset, scaled_circ*-1*0.7071+offset] ]
		self.MIN = -0. 
		self.MAX = 1.0 
		stddev_scale = [.02, .02, .02, .02, .02, .02, .02, .02]

		gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))
		self.data_centres = gmm.sample(sample_shape=(int(500*self.N_centers)))
		return gmm.sample(sample_shape=(int(500*self.batch_size.numpy())))

	def dataset_gmm8(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)

		if self.loss == 'RBF':
			self.center_dataset = tf.data.Dataset.from_tensor_slices((self.data_centres))
			self.center_dataset = self.center_dataset.shuffle(4)
			self.center_dataset = self.center_dataset.batch(self.N_centers)
			self.center_dataset = self.center_dataset.prefetch(5)
		return train_dataset




