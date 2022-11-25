from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from absl import app
from absl import flags
import json
import glob
from tqdm.autonotebook import tqdm
import shutil
from cleanfid import fid
import ot

import tensorflow_probability as tfp
tfd = tfp.distributions

##FOR FID
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
import scipy as sp
from numpy import iscomplexobj
from numpy.linalg import norm as norml2

from ops import *

class GAN_Metrics():

	def __init__(self):

		self.W22_flag = 0
		self.GradGrid_flag = 0
		self.metric_counter_vec = []

		if 'W22' in self.metrics:				
			self.W22_flag = 1
			self.W22_vec = []

			if self.data in ['g1', 'g2', 'gN']:
				self.W22_steps = 10
			else:
				self.W22_flag = 1
				self.W22_steps = 50
				print('W22 is not an accurate metric on this datatype')
		
		if 'GradGrid' in self.metrics:
			if self.data in ['g2', 'gmm8']:
				self.GradGrid_flag = 1
				self.GradGrid_steps = 50
			else:
				print("Cannot plot Gradient grid. Not a 2D dataset")



	def eval_metrics(self):
		update_flag = 0

		if self.W22_flag and ((self.total_count.numpy()%self.W22_steps == 0 or self.total_count.numpy() <= 10 or self.total_count.numpy() == 1)  or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_W22()
			if self.mode != 'metrics':
				np.save(self.metricpath+'W22.npy',np.array(self.W22_vec))
				self.print_W22()


		if self.GradGrid_flag and ((self.total_count.numpy()%self.GradGrid_steps == 0 or self.total_count.numpy() == 1) or self.mode == 'metrics'):
			update_flag = 1
			self.print_GradGrid()

		if self.res_flag and update_flag:
			self.res_file.write("Metrics avaluated at Iteration " + str(self.total_count.numpy()) + '\n')


	def update_W22(self):
		if self.data in ['g2']:
			self.eval_W22(self.reals,self.fakes)
		else:
			self.estimate_W22(self.reals,self.fakes)

	def eval_W22(self,act1,act2):
		mu1, sigma1 = act1.numpy().mean(axis=0), cov(act1.numpy(), rowvar=False)
		mu2, sigma2 = act2.numpy().mean(axis=0), cov(act2.numpy(), rowvar=False)
		ssdiff = np.sum((mu1 - mu2)**2.0)
		# calculate sqrt of product between cov
		if self.data not in ['g1', 'gmm2']:
			covmean = sqrtm(sigma1.dot(sigma2))
		else:
			covmean = np.sqrt(sigma1*sigma2)
		# check and correct imaginary numbers from sqrt
		if iscomplexobj(covmean):
			covmean = covmean.real
		# calculate score
		if self.data not in ['g1', 'gmm2']:
			self.W22_val = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
		else:
			self.W22_val = ssdiff + sigma1 + sigma2 - 2.0 * covmean
		self.W22_vec.append([self.W22_val, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final W22 score - "+str(self.W22_val))
			self.res_file.write("Final W22 score - "+str(self.W22_val))

		if self.res_flag:
			self.res_file.write("W22 score - "+str(self.W22_val))
		return

	def estimate_W22(self,target_sample, gen_sample, q=2, p=2):
		target_sample = tf.cast(target_sample, dtype = 'float32').numpy()
		gen_sample = tf.cast(gen_sample, dtype = 'float32').numpy()
		target_weights = np.ones(target_sample.shape[0]) / target_sample.shape[0]
		gen_weights = np.ones(gen_sample.shape[0]) / gen_sample.shape[0]

		x = target_sample[:, None, :] - gen_sample[None, :, :]

		M = tf.norm(x, ord=q, axis = 2)**p / p
		# print(target_sample.shape, gen_sample.shape, M.shape)
		T = ot.emd2(target_weights, gen_weights, M.numpy())
		self.W22_val = W = ((M.numpy() * T).sum())**(1. / p)

		self.W22_vec.append([self.W22_val, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final W22 score - "+str(self.W22_val))
			self.res_file.write("Final W22 score - "+str(self.W22_val))

		if self.res_flag:
			self.res_file.write("W22 score - "+str(self.W22_val))
		return


	def print_W22(self):
		path = self.metricpath
		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.W22_vec)[:,0])
		locs = list(np.array(self.W22_vec)[:,1])
		

		with PdfPages(path+'W22_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = r'$\mathcal{W}^{2,2}(p_d,p_g)$ Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)


	def print_GradGrid(self):

		path = self.metricpath + str(self.total_count.numpy()) + '_'

		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size": 12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		
		from itertools import product as cart_prod

		x = np.arange(self.MIN,self.MAX+0.1,0.1)
		y = np.arange(self.MIN,self.MAX+0.1,0.1)

		# X, Y = np.meshgrid(x, y)
		prod = np.array([p for p in cart_prod(x,repeat = 2)])
		# print(x,prod)

		X = prod[:,0]
		Y = prod[:,1]

		# print(prod,X,Y)
		# print(XXX)

		with tf.GradientTape() as disc_tape:
			prod = tf.cast(prod, dtype = 'float32')
			disc_tape.watch(prod)
			if self.loss == 'RBF':
				d_vals = self.discriminator_RBF(prod,training = False)
			else:
				d_vals = self.discriminator(prod,training = False)
		grad_vals = disc_tape.gradient(d_vals, [prod])[0]

		#Flag to control normalization of D(x) values for printing on the contour plot
		Normalize_Flag = False
		try:
			# print(d_vals[0])
			
			if Normalize_Flag and ((min(d_vals[0]) <= -2) or (max(d_vals[0]) >= 2)):
				### IF NORMALIZATION IS NEEDED
				d_vals_sub = d_vals[0] - min(d_vals[0])
				d_vals_norm = d_vals_sub/max(d_vals_sub)
				d_vals_norm -= 0.5
				d_vals_norm *= 3
				d_vals_new = np.reshape(d_vals_norm,(x.shape[0],y.shape[0])).transpose()
			else:
				### IF NORMALIZATION IS NOT NEEDED
				d_vals_norm = d_vals[0]
				d_vals_new = np.reshape(d_vals_norm,(x.shape[0],y.shape[0])).transpose()
		except:
			d_vals_new = np.reshape(d_vals,(x.shape[0],y.shape[0])).transpose()
		# print(d_vals_new)
		dx = grad_vals[:,1]
		dy = grad_vals[:,0]
		# print(XXX)
		n = -1
		color_array = np.sqrt(((dx-n)/2)**2 + ((dy-n)/2)**2)

		with PdfPages(path+'GradGrid_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=self.MIN,top=self.MAX)
			ax1.quiver(X,Y,dx,dy,color_array)
			ax1.scatter(self.reals[:1000,0], self.reals[:1000,1], c='r', linewidth = 1, label='Real Data', marker = '.', alpha = 0.1)
			ax1.scatter(self.fakes[:1000,0], self.fakes[:1000,1], c='g', linewidth = 1, label='Fake Data', marker = '.', alpha = 0.1)
			pdf.savefig(fig1)
			plt.close(fig1)

		with PdfPages(path+'Contour_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			ax1.contour(x,y,d_vals_new,15,linewidths = 1.0, alpha = 0.6 )
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.75)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.75)
			# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
			pdf.savefig(fig1)
			plt.close(fig1)



		

