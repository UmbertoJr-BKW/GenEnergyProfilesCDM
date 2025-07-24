import torch
import torch.nn as nn

# Define the Diffusion Model 
class Model(nn.Module):
    """
    It sets the device where to train, the noise scheduler, the Denoise Deep Learning Architecture, and other useful parameters saved in opt
    """
    def __init__(self, device, noise_schedule, dl_architecture, normalized=False):
        super().__init__()
        self.device = device
        self.schedule = noise_schedule
        self.denoiser = dl_architecture
        self.to(device=self.device)
        self.normalized = normalized
        

    def forward(self, x, idx=None, get_target=False):
        idx = torch.randint(0, len(self.schedule.alpha_bars), (x.size(0), 1)).to(device = self.device)
            
        # perturb random index and calculate x_tilde
        used_alpha_bars = self.schedule.alpha_bars[idx].unsqueeze(1)
        epsilon = torch.randn_like(x)
        x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
        
        if self.normalized:
            x_tilde = x_tilde / (x_tilde.std(dim=(1)).unsqueeze(1) + 1e-8)
            
        # Pass x_tilde and idx through the backbone module
        output = self.denoiser(x_tilde.float(), idx)

        return (output, epsilon) if get_target else output

    
# Define the DiffusionProcess class
class Sampler():

    # Initialize the class
    def __init__(self, beta_t, alpha_t, alphabar_t, diffusion_fn, device, shape, normalized=False):
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
        self.betas = torch.from_numpy(beta_t).to(device)
        self.alphas = torch.from_numpy(alpha_t).to(device)
        self.T = len(beta_t)
        self.alpha_bars = torch.from_numpy(alphabar_t).to(device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        
        self.shape = shape
        self.diffusion_fn = diffusion_fn.to(device)
        self.device = device
        self.normalized = normalized
    
    def _one_diffusion_step(self, x):
        '''
        
        Parameters:
        x: torch.Tensor
            The input tensor for the diffusion process 
        '''
        # Perform one diffusion step
        for idx in reversed(range(self.T)):
            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx] + 1e-9) * self.betas[idx])
            
            idx_t = torch.Tensor([int(idx) for _ in range(x.size(0))]).to(device = self.device).long()
            
            idx_t = idx_t.unsqueeze(-1)
            
            if self.normalized:
                x = x / (x.std(dim=(1)).unsqueeze(1) + 1e-8)
                
            predict_epsilon = self.diffusion_fn(x, idx_t)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)
            x = mu_theta_xt + sqrt_tilde_beta * noise
            yield x

    @torch.no_grad()
    def sampling(self, sampling_number, only_final=False):
        # Perform sampling from the diffusion process
        sample = torch.randn([sampling_number,*self.shape]).to(device = self.device).squeeze()
        sample = sample.unsqueeze(-1)
        sampling_list = []

        final = None
        for idx, sample in enumerate(self._one_diffusion_step(sample)):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return final if only_final else torch.stack(sampling_list)