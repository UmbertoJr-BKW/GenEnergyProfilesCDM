import os
import torch
import numpy as np


_MODELS = {}

def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def create_score_model(config):
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    return score_model

def get_model_fn(model, train=False):
    def model_fn(x, labels):
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)
    return model_fn


def get_score_fn(sde, model, train=False):
    model_fn = get_model_fn(model, train=train)
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, t):
            labels = t * 999
            score = model_fn(x, labels)
            std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            score = -score / std[:, None]
            return score

    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t):
            labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            score = model_fn(x, labels)
            return score
    return score_fn

def get_conditional_model_fn(model, train=False):
    def model_fn(h, x, labels):

        if not train:
            model.eval()
            return model(h, x, labels)
        else:
            model.train()
            return model(h, x, labels)

    return model_fn


def get_conditional_score_fn(sde, model, train=False):
    model_fn = get_conditional_model_fn(model, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(h, x, t):
            labels = t * 999
            score = model_fn(h, x, labels)
            std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            score = -score / std[:, None]
            return score
    return score_fn

def get_sigmas(config):
    sigmas = np.exp(
        np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))
    return sigmas


def save_AE_checkpoint(ckpt_dir, state):
    saved_state = {
        'encoder': state['encoder'].state_dict(),
        'decoder': state['decoder'].state_dict(),
        'opt_e' : state['opt_e'].state_dict(),
        'opt_r' : state['opt_r'].state_dict()
    }
    torch.save(saved_state, ckpt_dir)

    

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['conditional_model'].load_state_dict(loaded_state['conditional_model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state

def restore_AE_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['encoder'].load_state_dict(loaded_state['encoder'], strict=False)
    state['decoder'].load_state_dict(loaded_state['decoder'], strict=False)
    #state['opt_e'].load_state_dict(loaded_state['opt_e'])
    #state['opt_r'].load_state_dict(loaded_state['opt_r'])
    return state
