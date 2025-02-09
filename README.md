# Machine Learning Applied to Relativistic Heavy Ion Collision

Flow harmonics characterize the anisotropic expansion of the QGP and help in understanding the fundamental aspects of quantum chromodynamics (QCD). Evaluating flow harmonics (arXiv:2304.00336) in heavy-ion collisions is crucial because they provide insights into the properties of the quark-gluon plasma (QGP), a state of matter that existed in the early universe.

## Data creation/manipulation
The Data folder is where the dataset is created. Starting from the MUSIC RHIC text files, the heat maps are created based on the transversium momentum and the azimuthal angle of all particles in a event, which each event correspond to a single image in the dataset. The "Eos_inputrework_fast.py" is how this data ins transformed and "pixel_analysis.py" is a self prove of how the heat maps trully represents the original data. 

## Data validation
To validate the data provided, the YiLunDu equation of state classification paper (arXiv:1910.11530) was reproduced with 99.75% and 99.09% accuracies of train and validation datasets respectively. This reproduce was done in ^Eos_network.py".

## Flow estimation using Convolutional Neural Networks
"flow_idea_tt.py" and "flow_idea_ft.py" was so far the best I could achieve with regular CNNs. As far as it was possible for me to test, this method couldn't handle the task. The network couldn't return something simply better than the average of the test set. I also tried using grid search instead of regular hyperparametrization, but computational power wasnt enough to make this avaiable.

## Flow estimation using Variational Autoencoders
Before going with the method choosed (CVAE, based on the results found in arXiv:1909.06296), VAE was first applied being it a simpler version so, easier and faster to apply. Unexpectedly, even though VAE did not returned a very good result, it does could find some patterns in data, since it returned some results. As far I could test it, the average R² ("accuracy") of estimating the lowest order therms of flow harmonics was 59.93% ("vae_to_fllow_3.py"). The studies continues using, finally, CVAE.

## Flow estimation using Conditional Variational Autoencoders
Working on it ("cvae_to_flow_st.py")
