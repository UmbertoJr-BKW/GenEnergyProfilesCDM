import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import tensorflow as tf

from auto_encoder import EncoderLSTM, DecoderLSTM

from tqdm.notebook import tqdm
import time

from torch.utils import tensorboard

from utils import save_AE_checkpoint

def train_AutoEncoder(opt, dataset, device):
    
    nete = EncoderLSTM(input_size=opt.z_dim, 
                       hidden_dim=opt.hidden_dim, 
                       batch_size=opt.batch_size, 
                       n_layers=opt.num_layer,
                       device=device).to(device)
    #nete = torch.nn.DataParallel(nete, device_ids=[0, 1])
    
    netr = DecoderLSTM(hidden_dim=opt.hidden_dim,
                       output_size=opt.z_dim,
                       batch_size=opt.batch_size,
                       n_layers=opt.num_layer,
                       forecasting_horizon=opt.seq_len,
                       device=device).to(device)
    #netr = torch.nn.DataParallel(netr, device_ids=[0, 1])
    
    #input_data = torch.tensor(ori_data, dtype=torch.float32).to(device)
    nete.train()
    netr.train()
    optimizer_e = optim.Adam(nete.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_r = optim.Adam(netr.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    
    # setup the tensorboard writer and the directory to save the models
    print(f"{opt.module} {opt.num_layer}-layers {opt.hidden_dim}-Hdim  {opt.batch_size}-bs {opt.norm}-norm {opt.alpha_norm}-an")
    

    
    tb_dir = f"./AE_trained/tensorboard/{opt.module}/{opt.num_layer}-l/{opt.hidden_dim}-Hdim/{opt.batch_size}-bs{['/n-norm',f'/y-norm/{opt.alpha_norm}-an'][opt.norm]}/"
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    checkpoint_dir = f"./AE_trained/checkpoints/{opt.module}/{opt.num_layer}-l/{opt.hidden_dim}-Hdim/{opt.batch_size}-bs{['/n-norm',f'/y-norm/{opt.alpha_norm}-an'][opt.norm]}/"
    tf.io.gfile.makedirs(checkpoint_dir)

    # get the loss function used for the auto-encoder
    ER_step_fn = get_AE_step_fn(opt)

    #ckpt = sorted([ int(i[11:-4]) for i in os.listdir(checkpoint_dir)])[-1]
    ckpt = 0
    ckpt_dir = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}.pth")

    state = {'encoder':nete, 'decoder':netr, 'opt_e':optimizer_e, 'opt_r':optimizer_r}
    #state = restore_ER_checkpoint(ckpt_dir, state, device=device) 
    
    dataset_t = SmartMetersDataset(dataset)
    
    # Train the model
    ol = tqdm(range(opt.n_epochs), desc="Epoch Loop")
    step = 0
    for i in ol:
        data_trainer = DataLoader(dataset_t, shuffle=True, batch_size=opt.batch_size)
        il = tqdm(data_trainer, desc="Steps", leave=False)
        for x in il:
            ol.set_description(f"Epoch: {i}")
            il.set_description(f"Step: {i}")
            
            ER_loss, max_value, min_value, l_mean, l1_regularization = ER_step_fn(state, x.unsqueeze(-1).type(torch.float32).to(device))
            step += 1
            il.set_postfix(loss=float(l_mean.detach().cpu()))
            if step % 50 == 0:
                writer.add_scalar("training_ER_loss-mean", l_mean, step)
                writer.add_scalar("training_ER_max", max_value, step)
                writer.add_scalar("training_ER_min", min_value, step)
                writer.add_scalar("sparsity", l1_regularization, step)
                writer.add_scalar("actual-loss", ER_loss, step)


            if (step+1) % 3000 == 0:
                # Save the checkpoint.
                save_step = (step+1) // 10000
                save_step += ckpt
                save_AE_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
            


def get_AE_loss_fn(opt):
    def loss_fn(encoder, decoder, X, alpha=opt.alpha_norm, regularization=opt.norm):  # alpha is the regularization coefficient  
        encoder_hidden = encoder.init_hidden(batch_size=X.shape[0])
        output_encoder, hidden_embeddings = encoder.forward(X, encoder_hidden)
        input_decoder = hidden_embeddings[0][-1, :, :].unsqueeze(0).permute(1, 0, 2)
        X_tilde, decoder_hidden, _ = decoder.forward(decoder_input=input_decoder, encoder_hidden=hidden_embeddings)
    
        losses = torch.square(X-X_tilde)      
        l_mean =  losses.mean()
        
        hidden_representation = torch.cat([i for i in hidden_embeddings])
        l1_regularization = torch.norm(hidden_representation, 1)  
        if regularization:
            loss = l_mean + alpha * l1_regularization  
        else:
            loss = l_mean
        return loss, hidden_representation.max(), hidden_representation.min(), l_mean, l1_regularization
    
    return loss_fn  


def get_AE_step_fn(opt):
    AE_fn = get_AE_loss_fn(opt)

    def step_fn(state, X):
        nete = state['encoder']
        netr = state['decoder']
        optimizer_e = state['opt_e']
        optimizer_r = state['opt_r']
        nete.train()
        netr.train()

        optimizer_e.zero_grad()
        optimizer_r.zero_grad()
        loss, max_val, min_val, l_mean, l1_regularization = AE_fn(nete, netr, X)
        loss.backward(retain_graph=True)
        optimizer_e.step()
        optimizer_r.step()
        return loss, max_val, min_val, l_mean, l1_regularization

    return step_fn

            
            


class SmartMetersDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        time_series = self.dataset[idx, :]
        return time_series