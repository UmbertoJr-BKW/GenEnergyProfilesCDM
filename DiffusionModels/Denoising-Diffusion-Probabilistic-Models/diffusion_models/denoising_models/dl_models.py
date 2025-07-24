import math 
import torch
import torch.nn as nn

# Multi Layer Perceptron
class MLP(nn.Module):
    def __init__(self, n_steps, input_dim = 2, number_diffusion_steps=100):
        super().__init__()
        self.n_steps = n_steps
        self.input_dim = input_dim
        
        # Linear model 1 to map the input dimension to 256 hidden embeddings
        self.linear_model1 = nn.Sequential(
            nn.Linear(input_dim * n_steps, 256),
            nn.Dropout(0.2),
            nn.GELU()
        )
        
        # Embedding layer for conditioning on time t
        self.embedding_layer = nn.Embedding(number_diffusion_steps, 256)

        # Linear model 2 to join the contidioning on time and the input time series
        self.linear_model2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(512, input_dim * n_steps),
        )
        
        self.name = "MLP"
        self.num_layers = 5
        self.hdim = hidden_dim
        print(f'The {self.name} created has {count_parameters(self)} trainable parameters')  
    
    def forward(self, x, idx):
        x = torch.reshape(x, (-1, self.n_steps * self.input_dim))
        # Pass input through linear model 1, add the embedding layer, and pass through linear model 2
        x = self.linear_model2(self.linear_model1(x) + self.embedding_layer(idx).squeeze(1))
        x = torch.reshape(x, (-1, self.n_steps, self.input_dim))
        return x


# LSTM
class LSTM(nn.Module):
    def __init__(self, n_steps, input_dim=1, hidden_dim=128, num_layers=3, T=100):    
        super().__init__()    
        
        # Linear Layer to increase dimensionality from 1 to 128  
        self.linear = nn.Linear(input_dim, hidden_dim)  
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Embedding Layer    
        self.embedding_layer = nn.Embedding(T, hidden_dim)  # Embedding dimension is now 128  
        self.ln2 = nn.LayerNorm(hidden_dim)
           
        self.lstm = nn.LSTM(
            input_size = 2*hidden_dim, 
            hidden_size = 2*hidden_dim, 
            num_layers=num_layers, 
            bias=False,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        self.num_layers = num_layers   
        
        # Decoder Layer    
        self.decoder = nn.Sequential(
            nn.Linear(4*hidden_dim, 512),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.Dropout(0.2),
            nn.GELU(),

            nn.Linear(512, input_dim),
        )
        
        self.name = "LSTM"
        self.hdim = hidden_dim
        print(f'The {self.name} created has {count_parameters(self)} trainable parameters')  

        

        
        
    def forward(self, x, idx):  
        to_print = False
        if to_print:
            print(f"input shape: {x.shape} - {idx.shape}")  
        x = self.linear(x)
        x = self.ln1(x)
        if to_print:
            print(f"after linear shape: {x.shape}")  
        idx = idx.long()  
        emb = self.embedding_layer(idx) 
        if to_print:
            print(emb.shape)
        emb = emb.repeat((1, x.shape[1], 1 ))
        if to_print:
            print(emb.shape)
        x = torch.cat((x, emb), dim=-1) 
        x, (h, c) = self.lstm(x) 
        if to_print:
            print(f"after lstm shape: {x.shape}")  
        x = self.decoder(x)
        if to_print:
            print(f"output shape: {x.shape}")  
        return x
    

# Transformer
class Transformer(nn.Module):    
    def __init__(self, n_steps, input_dim=1, hidden_dim=128, nhead=8, num_layers=6, T=100):    
        super().__init__()    
        
        # Linear Layer to increase dimensionality from 1 to 128  
        self.linear = nn.Linear(input_dim, hidden_dim)  
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Embedding Layer    
        self.embedding_layer = nn.Embedding(T, hidden_dim)  # Embedding dimension is now 128  
        self.ln2 = nn.LayerNorm(hidden_dim)
            
        # Positional Encoding    
        self.pos_encoder = PositionalEncoding(hidden_dim)  # Input dimension is now 128  
    
        # Transformer Encoder Layers        
        encoder_layers = nn.TransformerEncoderLayer(
            2*hidden_dim, 
            nhead,
            activation="gelu",
            batch_first=True,
            dropout=0.1
        )    
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers
        ) 
        self.num_layers = num_layers   
        self.name = "Transformer"
        self.hdim = hidden_dim
        print(f'The {self.name} created has {count_parameters(self)} trainable parameters')  
        
        # Decoder Layer    
        self.decoder = nn.Linear(2*hidden_dim, input_dim)  # Input dimension is now 128 
     
        # initialize the weigths    
        self.init_weights()

        
    def init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight)  
        nn.init.kaiming_normal_(self.decoder.weight)  
        initrange = 0.1  
        self.embedding_layer.weight.data.uniform_(-initrange, initrange)  
        for module in self.transformer_encoder.modules():  
            if isinstance(module, nn.Linear):  
                nn.init.kaiming_normal_(module.weight)  
            elif isinstance(module, nn.LayerNorm):  
                module.bias.data.zero_()  
                module.weight.data.fill_(1.0)  

        
        
    def forward(self, x, idx):  
        to_print = False
        if to_print:
            print(f"input shape: {x.shape} - {idx.shape}")  
        x = self.linear(x)
        x = self.ln1(x)
        if to_print:
            print(f"after linear shape: {x.shape}")  
        idx = idx.long()  
        emb = self.embedding_layer(idx) 
        if to_print:
            print(emb.shape)
        emb = emb.repeat((1, x.shape[1], 1 ))
        if to_print:
            print(emb.shape)
        if to_print:
            print(f"after embedding shape: {x.shape}")  
        x = self.pos_encoder(x)
        if to_print:
            print(f"after pos encoder shape: {x.shape}")  
        x = torch.cat((x, emb), dim=-1) 
        x = self.transformer_encoder(x) 
        if to_print:
            print(f"after transformer shape: {x.shape}")  
        x = self.decoder(x)
        if to_print:
            print(f"output shape: {x.shape}")  
        return x
    


# Positional Encoding class to incorporate the sequence information  
class PositionalEncoding(nn.Module):  
  
    def __init__(self, d_model, dropout=0.1, max_len=96):  
        super(PositionalEncoding, self).__init__()  
        self.dropout = nn.Dropout(p=dropout)  
        
        # Create positional encodings  
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)    
        self.register_buffer('pe', pe)  
  
    def forward(self, x):  
        # Add positional encodings to the input 
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)  
    

def count_parameters(model):  
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  
