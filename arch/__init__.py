from absl import flags
import os, sys, time, argparse

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if FLAGS.gan == 'WGAN':
	if FLAGS.loss == 'RBF' and FLAGS.topic == 'RBFCoulombGAN':
		from .arch_RBF import *
	else:
		from .arch_base import *