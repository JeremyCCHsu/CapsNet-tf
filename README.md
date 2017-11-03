A partial implementation of [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)  
Currently this repo contains only the experiments on MNIST.  

The CapsuleNet has an amazingly simple architecture. In my opinion, they have at least 2 special features:
1. Iterative attention mechanism (they call it "dynamic routing).  
2. Vector representation of concepts/objects, with the magnitude representing the probibility of an object's existence.


# Dependency
Python 3.5  
Tensorflow 1.4  

# Usage
Run `python capsule.py` and use `tensorboard --logdir [logdir/train/datetime-Capsule]` to see the results.


# Results 
Their figure showing the result of dimension pertubation.  
![their results](img/fig4.png)  

<br>

Each column is a dimension of the Digit Capsule.  
Each row is a perturbation [-.25, 0.25] with 0.05 increment.  

![reconstruction from latent space with perturbation](img/img0.png)
![reconstruction from latent space with perturbation](img/img1.png)
![reconstruction from latent space with perturbation](img/img2.png)
![reconstruction from latent space with perturbation](img/img3.png)
![reconstruction from latent space with perturbation](img/img4.png)
![reconstruction from latent space with perturbation](img/img5.png)
![reconstruction from latent space with perturbation](img/img6.png)
![reconstruction from latent space with perturbation](img/img7.png)
![reconstruction from latent space with perturbation](img/img8.png)
![reconstruction from latent space with perturbation](img/img9.png)  
Indeed, there are always some dimensions that represent orientation and thickness.


The image, reconstruction, and latent representation of an input of digit 8.  
<img src="img/true8.png" width=100>
<img src="img/reconst8.png" width=100>
<img src="img/act8.png" height=200>  
The 8th row is highly activated, so the prediction is naturally 8 (note that the competing digit is 3 but the magnitude is apparently lower). In addition, the highly activated dimensions (7 & 8) represent thickness and orientation. (use `tensorboard` to view the results above.)  

# Difference
1. I rescaled the input to [-1, 1] and used `tanh` as the output non-linearity of the reconstruction net.  
2. Not all hyper-parameters were specified in their paper.
