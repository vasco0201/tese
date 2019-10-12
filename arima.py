import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO

# specify training data
data = pd.read_csv("freeway_data/freeway_data2.csv", usecols=[1], engine='python')
dataset = data.values
dataset = dataset.astype('float32')

train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

#efine model
my_order = (3, 0, 2)
my_seasonal_order = (0, 1, 0, 96)


# finding best ar an ma
#esDiff = sm.tsa.arma_order_select_ic(tra, max_ar=7, max_ma=7, ic='aic', trend='c')
#print('ARMA(p,q) =',resDiff['aic_min_order'],'is the best.')
# define model
model =  sm.tsa.statespace.SARIMAX(train, order=my_order, seasonal_order=my_seasonal_order)
model_fit = model.fit()