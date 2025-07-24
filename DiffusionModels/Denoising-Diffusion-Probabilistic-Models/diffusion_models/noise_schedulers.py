import torch
import numpy as np

def cosine_schedule(T):
    s = 0.02
    t_vec = np.arange(T+1)
    f_t = np.cos([(t/T + s)/(1+s) * np.pi/2 for t in t_vec])**2
    alphabar_t = np.clip(f_t/ f_t[0], 0, 0.99999)
    beta_t = np.array([min(1 - a_t/a_t1, 0.999) for a_t, a_t1 in zip(alphabar_t[1:], alphabar_t[:-1])])
    alpha_t = 1 - beta_t
    return beta_t, alpha_t, alphabar_t[:-1]

def linear_schedule(beta_min, beta_max, T):
    beta_t = np.linspace(beta_min, beta_max, T)
    alpha_t = 1 - beta_t
    alphabar_t = np.cumprod(alpha_t)
    return beta_t, alpha_t, alphabar_t
     
    
class Scheduler():
    def __init__(self, device, beta, alpha, alpha_bar):
        self.betas = torch.from_numpy(beta).to(device=device)
        self.alphas = torch.from_numpy(alpha).to(device=device)
        self.alpha_bars = torch.from_numpy(alpha_bar).to(device=device)
        self.prev_alpha_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        

def linear_on_alphabars(T):
    alphabar_t = np.linspace(1e-12, 1-1e-12, T+1)[::-1]
    beta_t = np.array([min(1 - a_t/a_t1, 0.999) for a_t, a_t1 in zip(alphabar_t[1:], alphabar_t[:-1])])
    alpha_t = 1 - beta_t
    return beta_t, alpha_t, alphabar_t[:-1]


def mix_cos_lin(T):
    s = 0.02
    t_vec = np.arange(T+1)
    f_t = np.cos([(t/T + s)/(1+s) * np.pi/2 for t in t_vec])**2
    alphabar_t_cos = np.clip(f_t/ f_t[0], 0, 0.99999)
    alphabar_t_lin = np.linspace(1e-12, 1-1e-12, T+1)[::-1]
    
    change_index = np.argmin(np.abs(alphabar_t_cos[:-10] - alphabar_t_lin[:-10]))
    
    alphabar_t = np.concatenate((alphabar_t_cos[:change_index], alphabar_t_lin[change_index:]), axis=0)
    
    beta_t = np.array([min(1 - a_t/a_t1, 0.999) for a_t, a_t1 in zip(alphabar_t[1:], alphabar_t[:-1])])
    alpha_t = 1 - beta_t
    return beta_t, alpha_t, alphabar_t[:-1]




def sigmoid_noiseschedule(T, start=-3, end=3, tau=1.0, clip_min=1e-9):
    t_vec = np.arange(T+1)
    v_sigmoid = np.vectorize(sigmoid_schedule, excluded=['start', 'end', 'tau', 'clip_min'])
    alphabar_t = v_sigmoid(t=t_vec/T, start=start, end=end, tau=tau, clip_min=clip_min)
    beta_t = np.array([min(1 - a_t/a_t1, 0.999) for a_t, a_t1 in zip(alphabar_t[1:], alphabar_t[:-1])])
    alpha_t = 1 - beta_t
    return beta_t, alpha_t, alphabar_t[:-1]
    
def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
    # A gamma function based on sigmoid function.
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.)

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def personalized_noise_schedule(T):
    t_vec = np.arange(T+1)
    v_sigmoid = np.vectorize(sigmoid_schedule, excluded=['start', 'end', 'tau', 'clip_min'])
    alphabar_t_sig = v_sigmoid(t=t_vec/T, start=-4, end=1, tau=0.7,  clip_min=1e-9)
    alphabar_t_lin = np.linspace(1e-12, 1-1e-12, T+1)[::-1]
    
    change_index = np.argmin(np.abs(alphabar_t_sig[10:-10] - alphabar_t_lin[10:-10]))
    
    alphabar_t = np.concatenate((alphabar_t_sig[:change_index+10], 
                                 alphabar_t_lin[change_index+10:]), axis=0)
    
    beta_t = np.array([min(1 - a_t/a_t1, 0.999) for a_t, a_t1 in zip(alphabar_t[1:], alphabar_t[:-1])])
    alpha_t = 1 - beta_t
    return beta_t, alpha_t, alphabar_t[:-1]