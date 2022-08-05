from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence
import math
import numpy as np 
import pandas as pd 

class DataGenerator(Sequence):
    def __init__(self,opt,
                   df:pd.DataFrame):
        self.df = df 
        self.opt = opt 
        self.timestamps = opt['timestamps']
        self.shuffle = opt['shuffle']
        self.batch_size = opt['batch_size'] 
        self.std = {timestamp : 0 for timestamp in opt['timestamps']}
        
    def __len__(self):
        return math.ceil(len(self.timestamps) / self.batch_size)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.timestamps))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            self.timestamps = list(np.array(self.timestamps)[self.indices])

    def subset(self,timestamp):
        if self.std[timestamp] ==0:
            input_array = self.df.loc[self.df['ymdhm']<timestamp].iloc[-self.opt['subset_length']:][self.opt['input_columns']].to_numpy()
            output_array = self.df.loc[self.df['ymdhm']==timestamp][self.opt['output_columns']].to_numpy()
            self.std[timestamp] = (input_array,output_array)
        else:
            (input_array,output_array) = self.std[timestamp]
        
        return input_array, output_array 
     
    
    def batch_subset(self,batch_timestamps):
        input_df = []
        output_df = [] 
        for timestamp in batch_timestamps:
            input_array, output_array = self.subset(timestamp)
            input_df.append(input_array)
            output_df.append(output_array)
        
        return np.array(input_df),np.array(output_df)

    def __getitem__(self, index):
        self.batch_timestamps = self.timestamps[index*self.batch_size:(index+1)*self.batch_size]
        input_df,output_df = self.batch_subset(self.batch_timestamps) 
        return input_df.astype(np.float16),output_df.squeeze().astype(np.float16)