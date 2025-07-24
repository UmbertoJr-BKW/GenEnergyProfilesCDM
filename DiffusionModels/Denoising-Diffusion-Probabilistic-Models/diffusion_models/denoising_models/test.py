import torch
from dl_models import MLP, LSTM, Transformer



# Create random example data for testing
n_steps = 96  
input_dim = 1
batch_size=32

input_data = torch.randn(batch_size, n_steps, input_dim)  
index = torch.randint(high=n_steps, size=(batch_size, 1))  

print(f"input shape {input_data.shape}")
print(f"index shape {index.shape}")
  
#### test of MLP
model = MLP(n_steps, input_dim)   
output = model(input_data, index)  
print(f"Output shape of model {model.name}: {output.shape}") 


#### test of LSTM
model = LSTM(n_steps, input_dim)   
output = model(input_data, index)  
print(f"Output shape of model {model.name}: {output.shape}") 


#### test of Transformer
model = Transformer(n_steps, input_dim)   
output = model(input_data, index)  
print(f"Output shape of model {model.name}: {output.shape}") 





