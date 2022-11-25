from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import tensorflow_probability as tfp
from matplotlib.backends.backend_pgf import PdfPages

import math
import tensorflow as tf
from absl import app
from absl import flags

from gan_topics import *

'''***********************************************************************************
********** Baseline WGANs ************************************************************
***********************************************************************************'''
### self.gan = 'WGAN' and self.topic = 'Base' and self.loss = 'GP', 'LP', 'ALP', 'R1', 'R2'.
class WGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):

		GAN_Base.__init__(self,FLAGS_dict)

		self.lambda_GP = 0.1 
		self.lambda_ALP = 10.0 
		self.lambda_LP = 0.1 
		self.lambda_R1 = 0.5
		self.lambda_R2 = 0.5

	#################################################################

	def create_optimizer(self):
		with tf.device(self.device):
			if  self.loss == 'ALP' :
				self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100, decay_rate=0.9, staircase=True)
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_schedule, self.beta1, self.beta2)
			else:
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)
			print("Optimizers Successfully made")	
		return	

	#################################################################

	def save_epoch_h5models(self):
		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
		self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return

	#################################################################

	def train_step(self,reals_all):
		for i in tf.range(self.Dloop):
			with tf.device(self.device):
				noise = self.get_noise([self.batch_size, self.noise_dims])
			self.reals = reals_all

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				
				self.fakes = self.generator(noise, training=True)

				self.real_output = self.discriminator(self.reals, training = True)
				self.fake_output = self.discriminator(self.fakes, training = True)
				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))

			if self.loss == 'base':
				wt = []
				for w in self.discriminator.get_weights():
					w = tf.clip_by_value(w, -0.1,0.1) #0.01 for [0,1] data, 0.1 for [0,10]
					wt.append(w)
				self.discriminator.set_weights(wt)
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	#################################################################

	def loss_base(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output) 

		self.D_loss = 1 * (-loss_real + loss_fake)
		self.G_loss = 1 * (loss_real - loss_fake)

	#################################################################

	def loss_GP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_GP * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty(self):
		if self.data in ['g2', 'gmm8', 'gN']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		diff = tf.cast(self.fakes,dtype='float32') - tf.cast(self.reals,dtype='float32')
		inter = tf.cast(self.reals,dtype='float32') + (alpha * diff)
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])[0]
		if self.data in ['g2', 'gmm8', 'gN']:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		else:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
		self.gp = tf.reduce_mean((slopes - 1.)**2)
		return 


	#################################################################

	def loss_R1(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_R1()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R1 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_R1(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py
		if self.data in ['g2', 'gmm8', 'gN']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		inter = tf.cast(self.reals,dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])[0]
		if self.data in ['g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_R2(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_R2()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R2 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_R2(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py
		if self.data in ['g2', 'gmm8', 'gN']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		inter = tf.cast(self.fakes,dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])[0]
		if self.data in ['g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_LP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.lipschitz_penalty()

		self.D_loss = -loss_real + loss_fake + self.lambda_LP * self.lp 
		self.G_loss = loss_real - loss_fake

	def lipschitz_penalty(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py

		self.K = 1
		self.p = 2

		if self.data in ['g2', 'gmm8', 'gN']:
			epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1], 0.0, 1.0)
		else:
			epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1, 1, 1], 0.0, 1.0)
		x_hat = epsilon * self.fakes + (1 - epsilon) * self.reals

		with tf.GradientTape() as t:
			t.watch(x_hat)
			D_vals = self.discriminator(x_hat, training = False)
		grad_vals = t.gradient(D_vals, [x_hat])[0]

		#### args.p taken from github as default p=2
		dual_p = 1 / (1 - 1 / self.p) if self.p != 1 else np.inf

		grad_norms = tf.norm(grad_vals, ord=dual_p, axis=1, keepdims=True)
		self.lp = tf.reduce_mean(tf.maximum(grad_norms - self.K, 0)**2)

	#################################################################

	def loss_ALP(self):
		
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.adversarial_lipschitz_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_ALP * self.alp 
		self.G_loss = 1 * (loss_real - loss_fake)


	def adversarial_lipschitz_penalty(self):
		def normalize(x, ord):
			return x / tf.maximum(tf.norm(x, ord=ord, axis=1, keepdims=True), 1e-10)
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py
		self.eps_min = 0.1
		self.eps_max = 10.0
		self.xi = 10.0
		self.ip = 1
		self.p = 2
		self.K = 5 #was 1. made 5 for G2 compares

		samples = tf.concat([self.reals, self.fakes], axis=0)
		if self.data in ['g2', 'gmm8', 'gN']:
			noise = tf.random.uniform([tf.shape(samples)[0], 1], 0, 1)
		else:
			noise = tf.random.uniform([tf.shape(samples)[0], 1, 1, 1], 0, 1)

		eps = self.eps_min + (self.eps_max - self.eps_min) * noise

		with tf.GradientTape(persistent = True) as t:
			t.watch(samples)
			validity = self.discriminator(samples, training = False)

			d = tf.random.uniform(tf.shape(samples), 0, 1) - 0.5
			d = normalize(d, ord=2)
			t.watch(d)
			for _ in range(self.ip):
				samples_hat = tf.clip_by_value(samples + self.xi * d, clip_value_min=-1, clip_value_max=1)
				validity_hat = self.discriminator(samples_hat, training = False)
				dist = tf.reduce_mean(tf.abs(validity - validity_hat))
				grad = t.gradient(dist, [d])[0]
				# print(grad)
				d = normalize(grad, ord=2)
			r_adv = d * eps

		samples_hat = tf.clip_by_value(samples + r_adv, clip_value_min=-1, clip_value_max=1)

		d_lp                   = lambda x, x_hat: tf.norm(x - x_hat, ord=self.p, axis=1, keepdims=True)
		d_x                    = d_lp

		samples_diff = d_x(samples, samples_hat)
		samples_diff = tf.maximum(samples_diff, 1e-10)

		validity      = self.discriminator(samples    , training = False)
		validity_hat  = self.discriminator(samples_hat, training = False)
		validity_diff = tf.abs(validity - validity_hat)

		alp = tf.maximum(validity_diff / samples_diff - self.K, 0)
		# alp = tf.abs(validity_diff / samples_diff - args.K)

		nonzeros = tf.greater(alp, 0)
		count = tf.reduce_sum(tf.cast(nonzeros, tf.float32))

		self.alp = tf.reduce_mean(alp**2)
		# alp_loss = args.lambda_lp * reduce_fn(alp ** 2)

	#####################################################################

	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError()
		loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))#mse(self.reals, self.reals_dec)
		self.AE_loss =  loss_AE_reals  

'''***********************************************************************************
********** WGAN ELEGANT WITH LATENT **************************************************
***********************************************************************************'''
### self.gan = 'WGAN' and self.topic = 'CoulombGAN' and self.loss = 'RBF'
class WGAN_CoulombGAN(GAN_Base, RBFSolver):

	def __init__(self,FLAGS_dict):

		GAN_Base.__init__(self,FLAGS_dict)

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		RBFSolver.__init__(self)

		self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.4e}', 2: f'{0:2.4e}'}
		self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining}  Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'

		self.first_iteration_flag = 1

		self.kernel_dimension = 3
		self.epsilon = 1.0

	def create_models(self):
		# with tf.device(self.device):
		self.total_count = tf.Variable(0,dtype='int64')
		self.generator = eval(self.gen_model)
		self.discriminator = eval(self.disc_model)
		self.discriminator_RBF = self.discriminator_model_RBF()

		print("Model Successfully made")
		print("\n\n GENERATOR MODEL: \n\n")
		print(self.generator.summary())
		print("\n\n DISCRIMINATOR MODEL: \n\n")
		print(self.discriminator.summary())
		print("\n\n DISCRIMINATOR RBF: \n\n")
		print(self.discriminator_RBF.summary())

		if self.res_flag == 1 and self.resume != 1:
			with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
				# Pass the file handle in as a lambda function to make it callable
				fh.write("\n\n GENERATOR MODEL: \n\n")
				self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
				self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
				fh.write("\n\n DISCRIMINATOR RBF: \n\n")
				self.discriminator_RBF.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):
		# with tf.device(self.device):
		self.lr_G_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=500, decay_rate=0.9, staircase=True)
		self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G) #Nadam or #SGD
		self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D) #Nadam or #SGD
		print("Optimizers Successfully made")
		return


	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							discriminator_RBF = self.discriminator_RBF, \
							total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator_RBF = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_RBF.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return





	def train_step(self,reals_all):

		for i in tf.range(self.Dloop):
			# with tf.device(self.device):
			noise = self.get_noise([self.batch_size, self.noise_dims])
			self.reals = reals_all
			if self.data in ['mnist']:
				self.reals += tfp.distributions.TruncatedNormal(loc=0., scale=0.1, low=-1.,high=1.).sample([self.batch_size, self.output_size, self.output_size, 1])

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

				self.fakes = self.generator(noise, training=True)

				with gen_tape.stop_recording(), disc_tape.stop_recording():
					self.pot_reals, self.pot_fakes = self.get_potentials(self.reals, self.fakes, self.kernel_dimension, self.epsilon)
		

				self.D_output_net_reals = self.discriminator(self.reals, training = True)
				self.D_output_net_fakes = self.discriminator(self.fakes, training = True)

				self.real_output_G = self.discriminator(self.reals, training = True)
				self.fake_output_G = self.discriminator(self.fakes, training = True)
				
				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))


			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def calculate_squared_distances(self,a, b):
		'''returns the squared distances between all elements in a and in b as a matrix
		of shape #a * #b'''
		na = tf.shape(a)[0]
		nb = tf.shape(b)[0]
		nas, nbs = list(a.shape), list(b.shape)
		a = tf.reshape(a, [na, 1, -1])
		b = tf.reshape(b, [1, nb, -1])
		a.set_shape([nas[0], 1, np.prod(nas[1:])])
		b.set_shape([1, nbs[0], np.prod(nbs[1:])])
		a = tf.tile(a, [1, nb, 1])
		b = tf.tile(b, [na, 1, 1])
		d = a-b
		return tf.reduce_sum(tf.square(d), axis=2)



	def plummer_kernel(self, a, b, dimension, epsilon):
		r = self.calculate_squared_distances(a, b)
		r += epsilon*epsilon
		f1 = dimension-2
		return tf.pow(r, -f1 / 2)


	def get_potentials(self, x, y, dimension, cur_epsilon):
		'''
		This is alsmost the same `calculate_potential`, but
			px, py = get_potentials(x, y)
		is faster than:
			px = calculate_potential(x, y, x)
			py = calculate_potential(x, y, y)
		because we calculate the cross terms only once.
		'''
		x_fixed = x
		y_fixed = y
		nx = tf.cast(tf.shape(x)[0], x.dtype)
		ny = tf.cast(tf.shape(y)[0], y.dtype)
		pk_xx = self.plummer_kernel(x_fixed, x, dimension, cur_epsilon)
		pk_yx = self.plummer_kernel(y, x, dimension, cur_epsilon)
		pk_yy = self.plummer_kernel(y_fixed, y, dimension, cur_epsilon)
		#pk_xx = tf.matrix_set_diag(pk_xx, tf.ones(shape=x.get_shape()[0], dtype=pk_xx.dtype))
		#pk_yy = tf.matrix_set_diag(pk_yy, tf.ones(shape=y.get_shape()[0], dtype=pk_yy.dtype))
		kxx = tf.reduce_sum(pk_xx, axis=0) / (nx)
		kyx = tf.reduce_sum(pk_yx, axis=0) / ny
		kxy = tf.reduce_sum(pk_yx, axis=1) / (nx)
		kyy = tf.reduce_sum(pk_yy, axis=0) / ny
		pot_x = kxx - kyx
		pot_y = kxy - kyy
		pot_x = tf.reshape(pot_x, [-1])
		pot_y = tf.reshape(pot_y, [-1])
		return pot_x, pot_y



	def loss_RBF(self):

		mse = tf.keras.losses.MeanSquaredError()

		# mse_real = mse(self.real_output_net, self.real_output)
		# mse_fake = mse(self.fake_output_net, self.fake_output)

		self.D_loss = mse(self.pot_reals, self.D_output_net_reals) + mse(self.pot_fakes, self.D_output_net_fakes) 
		# loss_fake = tf.reduce_mean(tf.minimum(self.fake_output,0))
		# loss_real = tf.reduce_mean(tf.maximum(self.real_output,0))
		loss_fake_G = tf.reduce_mean(self.fake_output_G)
		loss_real_G = tf.reduce_mean(self.real_output_G)

		# print(self.fake_output,loss_fake)
		# print(self.real_output,loss_real)

		self.G_loss = -1*loss_fake_G


		# self.D_loss = -1 * (-loss_real + self.alpha*loss_fake)
		# self.G_loss = -1 * (self.beta*loss_real - loss_fake)




'''***********************************************************************************
********** RBF-Coulomb GAN**************************************************
***********************************************************************************'''
### self.gan = WAE and self.topic = 'RBFCoulombGAN' and self.loss = 'RBF'
class WGAN_RBFCoulombGAN(GAN_Base, RBFSolver):

	def __init__(self,FLAGS_dict):

		GAN_Base.__init__(self,FLAGS_dict)

		''' Set up the RBF Series Solver common to PolyGAN and its WAE variant'''
		RBFSolver.__init__(self)

		self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.4e}', 2: f'{0:2.4e}'}
		self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining}  Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'

		self.first_iteration_flag = 1

	def create_models(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator_RBF = self.discriminator_model_RBF()

			print("Model Successfully made")
			print("\n\n GENERATOR MODEL: \n\n")
			print(self.generator.summary())
			print("\n\n DISCRIMINATOR RBF: \n\n")
			print(self.discriminator_RBF.summary())

			if self.res_flag == 1 and self.resume != 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR RBF: \n\n")
					self.discriminator_RBF.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):
		with tf.device(self.device):
			self.lr_G_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=500, decay_rate=0.9, staircase=True)
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G) #Nadam
		print("Optimizers Successfully made")
		return


	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							discriminator_RBF = self.discriminator_RBF, \
							total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator_RBF = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_RBF.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return


	def train_step(self,reals_all):
		with tf.device(self.device):
			noise = self.get_noise([self.batch_size, self.noise_dims])
		self.reals = reals_all

		with tf.GradientTape() as gen_tape:

			self.fakes = self.generator(noise, training=True)
				
			self.real_output = self.discriminator_RBF(self.reals, training = True)
			self.fake_output = self.discriminator_RBF(self.fakes, training = True)

			# print(self.real_output, self.fake_output)
			with gen_tape.stop_recording():
				Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
				self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])

				if self.first_iteration_flag:
					self.first_iteration_flag = 0 
					self.lamb = tf.constant(0.1)
					self.D_loss = self.G_loss = tf.constant(0)
					return

				self.find_lambda()
			self.divide_by_lambda()
			
			eval(self.loss_func)
			self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def loss_RBF(self):
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)

		self.D_loss = 1 * (-loss_real + loss_fake)
		if (2*self.rbf_m - self.rbf_n) >= 0:
			self.G_loss = loss_real - loss_fake
		elif (2*self.rbf_m - self.rbf_n) < 0:
			self.G_loss = -1*(loss_real - loss_fake)


