set -e
if [ "${CONDA_DEFAULT_ENV}" != "RBFCoulombGAN" ]; then
	echo 'You are not in the <RBFCoulombGAN> environment. Attempting to activate the RBFCoulombGAN environment via conda. Please run "conda activate RBFCoulombGAN" and try again if this fails. If the RBFCoulombGAN environment has not been installed, please refer the README.md file for further instructions.'
	condaActivatePath=$(which activate)
	source ${condaActivatePath} RBFCoulombGAN
fi

# An example of running RBFCoulombGAN and RBFCoulombGAN-WAE
# Data can be changed to mnist, cifar10, celeba and church when the ``gan`` flag is set to WAE
# Data can be changed to g2, gmm8, gN when the ``gan`` flag is set to WGAN
# Check paper for appropriate value for latent_dims
# We suggest setting rbf_m to ceil(n/2)


### Train call for RBFCoulombGAN on g2 learning 
python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RBFCoulombGAN' --mode 'train' --data 'g2' --noise_kind 'gaussian' --gan 'WGAN' --loss 'RBF' --arch 'dense' --latent_dims 2 --saver 1 --num_epochs 5 --res_flag 1 --lr_G 0.01 --lr_D 0.0 --paper 1 --batch_size '500' --metrics 'W22,GradGrid' --colab 0 --pbar_flag 1 --latex_plot_flag 0 --rbf_m 1 

### Train call for RBFCoulombGAN for gmm8 learning
python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RBFCoulombGAN' --mode 'train' --data 'gmm8' --noise_kind 'gaussian' --gan 'WGAN' --loss 'RBF' --arch 'dense' --latent_dims 2 --saver 1 --num_epochs 40 --res_flag 1 --lr_G 0.01 --lr_D 0.0 --paper 1 --batch_size '500' --metrics 'W22,GradGrid' --colab 0 --pbar_flag 1 --latex_plot_flag 0 --rbf_m 1 

