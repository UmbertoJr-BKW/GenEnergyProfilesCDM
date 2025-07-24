import torch
import pickle  
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler  
    

def reshape_dataframe(df):  
    scaler = MinMaxScaler(feature_range=(0, 1))  

    scaled_data = []  
    scaler_params = []  

    for idx, row in df.iterrows():  
        # Reshape your data  
        values = row.values.reshape(-1,1)

        # Compute the scaling parameters  
        scaler.fit(values)  

        # Save the min and max  
        scaler_params.append((scaler.data_min_, scaler.data_max_))  

        # Scale the data  
        scaled_values = scaler.transform(values)  

        # Append the scaled data  
        scaled_data.append(scaled_values)  

    # Convert the list of arrays into an array  
    scaled_data = np.stack(scaled_data, axis=0)  

    # Save scaler_params using pickle  
    with open("./outputs/scaler_params.pkl", "wb") as f:  
        pickle.dump(scaler_params, f)  
    return scaled_data


def restore_original_data(scaled_data):
    # Load scaler_params  
    with open("./outputs/scaler_params.pkl", "rb") as f:  
        scaler_params = pickle.load(f)  

    original_data = []
    for row in range(len(scaled_data)):
        data_max = scaler_params[row]  
        scaled_values = scaled_data[row, :]  
        original_values = scaled_values * (data_max * 1e-9)
        original_data.append(original_values)
    return original_data


def load_sm_data():
    transformed_df = pd.read_csv('./../data/sm_data.csv') 
    transformed_df['Id'] = transformed_df['Id'].astype(str)  
    transformed_df['Timestamp'] = pd.to_datetime(transformed_df['Timestamp'])  

    # Append the date to the Id before converting the 'Timestamp' to time  
    transformed_df['Id'] = transformed_df['Id'] + ' - ' + transformed_df['Timestamp'].dt.date.astype(str)  
    transformed_df['Timestamp'] = transformed_df['Timestamp'].dt.time  

    # Group by 'Id' and 'Timestamp' and calculate the mean of 'Energy_Consumption'  
    transformed_df = transformed_df.groupby(['Id', 'Timestamp'])['Energy_Consumption'].mean().reset_index()  

    # Pivot the table  
    transformed_df = transformed_df.pivot(index='Id', columns='Timestamp', values='Energy_Consumption') 

    # Drop the NaN values
    transformed_df = transformed_df.dropna()
    
    data = reshape_dataframe(transformed_df)
    return data


# Define the DataSet class for generating data samples
class DataSet(Dataset):      
    def __init__(self, tensor_input):  
        #df = df.diff()  
        self.tensor = tensor_input
        self.total_len = tensor_input.shape[0]  
    
    def __len__(self):    
        return self.total_len    
    
    def __getitem__(self, idx):      
        data = self.tensor[idx]
        return torch.from_numpy(data).float()