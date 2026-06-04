import numpy as np

def preprocessing(data, ):

  ''' This function is used to convert data to zero mean and unit variance

  Input Parameters:

  data: 1D pandas series

  '''

  processed_data = (data - np.mean(data, 0)) / np.std(data, 0)

  return processed_data