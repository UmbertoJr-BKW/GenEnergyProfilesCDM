import torch

# Define the DiffusionProcess class
class DiffusionProcess():

    # Initialize the class
    def __init__(self, beta_1, beta_T, T, diffusion_fn, device, shape):
        '''
        
        Parameters:
        beta_1: float
            The value of beta at time t = 1
        beta_T: float
            The value of beta at time t = T (the final time)    
        T: int
            The number of time steps
        diffusion_fn: function
            The function that defines the diffusion process
        device: str or torch.device object  
            The device to use for the diffusion process
        shape: tuple    
            The shape of the diffusion process
        '''
        self.betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.T = T
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, 
                                                           end=beta_T,
                                                           steps=T), 
                                        dim = 0).to(device = device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.shape = shape
        self.diffusion_fn = diffusion_fn
        self.device = device

    def _one_diffusion_step(self, x):
        '''
        
        Parameters:
        x: torch.Tensor
            The input tensor for the diffusion process 
        '''
        # Perform one diffusion step
        for idx in reversed(range(self.T)):
            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
            predict_epsilon = self.diffusion_fn(x, idx)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)
            x = mu_theta_xt + sqrt_tilde_beta * noise
            yield x

    @torch.no_grad()
    def sampling(self, sampling_number, only_final=False):
        # Perform sampling from the diffusion process
        sample = torch.randn([sampling_number,*self.shape]).to(device = self.device).squeeze()
        sampling_list = []

        final = None
        for idx, sample in enumerate(self._one_diffusion_step(sample)):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return final if only_final else torch.stack(sampling_list)
