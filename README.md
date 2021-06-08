# Adversarial Mirrored AutoEncoder

### Run command

```
python train.py --recon_loss_type='wasserstein' --ae_recon_loss_type='wasserstein' --use_penalty --spectral_norm=1 --anom_recon_lambda=5 --regularizer_lambda=1 --anom_pc=0.1 --dataset='cifar10' --expt_name='cifar_experiments' --normal_class="['ship']" --sampling
```
For OOD experiments run
```
python train.py --recon_loss_type='wasserstein' --ae_recon_loss_type='wasserstein' --use_penalty --spectral_norm=1 --anom_recon_lambda=5 --regularizer_lambda=1 --anom_pc=0.1 --dataset='cifar10' --expt_name='cifar_experiments' --ood_model --sampling 
```

### Main arguments
```
--dataset - Currently works with CIFAR10. 
--ood_model - Call this when you want to run OOD Anomaly Detection experiment.
--sampling - Turn sampling on/off using this
--atyp_selec_style - Which style sampling for anomalies. Options are 'inward|outward|sipple'
--interpolation_in_recon - Turns on simplex interpolation in training 
--corrup - Turn this on for corruption of training data. Use it along with --anom_pc to choose the level of corruption
--anom_pc - If used without --corrup, it results in Semi-supervised learning.
--normal_class - To select which class is normal in in-distribution AD experiments. Multiple classes can be selected.
--nz = Size of the latent space variable. 128 for CIFAR10 expts.
```

### Requirment
* Python3.7 Most recent pytorch version.
* wandb installed and initialized to visualize the trends

### Results
The best model is chosen based on AUC on validation data. The test AUC and other information are outputted as summary parameter on wandb.

Other option - the auc scores and reconstructions are outputted into a text file in logs folder.