# DDPM for Smart Meter's Consumption Measurements Modeling  
  
This project focuses on the application of the Denoising Diffusion Probabilistic Model (DDPM) for modeling energy consumption based on time series data obtained from CKW smart meters.  
  
### Installing  
  
Before running the project, install the necessary libraries and dependencies by executing the following command in your terminal:  
  
```sh  
pip install -r requirements.txt  
```

## Introduction to Denoising Diffusion Probabilistic Models (DDPM)  
  
Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that have gained traction due to their ability to model complex data distributions and generate high-quality samples. They belong to the family of diffusion-based generative models, which also include denoising diffusion probabilistic models (DDPM; [Ho et al. 2020](https://arxiv.org/abs/2006.11239)) and noise-conditioned score network (NCSN; [Yang & Ermon, 2020](https://arxiv.org/abs/1907.05600)),  
 
![Example of DM](./outputs/docs/DiffusionModels.png)

  
In a diffusion process, we start with a data point sampled from a real data distribution $x_0 \sim q(x)$, and add small amounts of Gaussian noise in incremental steps $t=1,...,T$. This produces a sequence of increasingly noisy samples $x_1,...,x_T$. As we add more noise $T \goes \infty$, the data sample gradually loses its distinguishable features and eventually becomes equivalent to an isotropic Gaussian distribution $x_T \approx z \sim N(0, 1)$. A remarkable property of this process is that we can sample a data point at any arbitrary time step using the reparameterization trick.

![Example of DM](./outputs/docs/diffusion-concept.png)
  
## Denoising with DDPM  
  
The real power of DDPM comes into play when we reverse the diffusion process. If we can sample from the noisy data and reverse the process, we can recreate the true sample from a Gaussian noise input. However, estimating the reverse conditional probability is not straightforward as it requires access to the entire dataset. Therefore, we need to learn a model to approximate these conditional probabilities.  
  
The reverse conditional probability becomes tractable when conditioned on the noise. By using Bayes’ rule, we can derive a relationship between the noisy sample, the true sample, and the noise. The mean and variance of the noisy sample can be parameterized, and this allows us to represent the noise in terms of the true sample and the noisy sample. This setup is similar to Variational Auto-Encoders (VAE), and we can use the variational lower bound to optimize the negative log-likelihood.  
  
![Example of DM](./outputs/docs/diffusion-example.png)
  
Training the diffusion model involves learning a neural network to approximate the conditioned probability distributions in the reverse diffusion process. The network is trained to predict the mean and variance of the Gaussian noise from the input at each time step. The loss term is parameterized to minimize the difference from the true sample.  
Interestingly, empirical observations show that training the diffusion model works better with a simplified objective that ignores a weighting term. 

![Example of DM](./outputs/docs/diffusion-algo.png)


## External Resources
More on DDPM can be found on the following [blog article](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

