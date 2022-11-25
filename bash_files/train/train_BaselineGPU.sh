set -e
if [ "${CONDA_DEFAULT_ENV}" != "RBFCoulombGAN" ]; then
	echo 'You are not in the <RBFCoulombGAN> environment. Attempting to activate the RBFCoulombGAN environment via conda. Please run "conda activate RBFCoulombGAN" and try again if this fails. If the RBFCoulombGAN environment has not been installed, please refer the README.md file for further instructions.'
	condaActivatePath=$(which activate)
	source ${condaActivatePath} RBFCoulombGAN
fi

# An example of running WGAN code on gmm8 for baselines:
# Losses can be chaned to GP, LP, ALP, R1 and R2.
# Data can be changed to g2, gmm8 

### Train call for Base WGAN-LP or GMMN-RBFG on gmm8
python ./gan_main.py  --run_id 'new' --resume 0 --GPU '2' --device '0' --topic 'Base' --mode 'train' --data 'gmm8' --latent_dims 1 --noise_kind 'gaussian' --gan 'WGAN' --loss 'LP' --arch 'dense' --saver 1 --num_epochs 50 --res_flag 1 --lr_G 0.002 --lr_D 0.0075 --Dloop 1 --paper 1 --batch_size '500' --Dloop 1 --metrics 'W22,GradGrid' --colab 0 --pbar_flag 1 --latex_plot_flag 0 

