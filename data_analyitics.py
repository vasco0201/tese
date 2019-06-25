import pandas as pd
import numpy as np
import copy
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
curr_dir = os.getcwd()

#initiate datasets
dataset = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\CT15Mn-150818_101018\\dados_camara.csv")#dados mais recentes
tomtom = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\tomtom_data.csv")
#dataset = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\periodic_data.csv") #dados periodicos gerados automaticamente
dataset = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\dados_old.csv") #dados mais antigos
dataset['unique_id'] = dataset.Zona.astype(str) + '_' + dataset.ID_Espira.astype(str)
dataset['unique_id'] = dataset['unique_id'].str.lower()

sensor_id = input("Input the area code followed by the loop sensor id: (AREA_ctID) \n Ex : 4_ct3 corresponds to area 4,sensor 3)")
chosen_date = input("Desired date: (DD-M-YYYY) ")

dataset = dataset.loc[(dataset.unique_id == str(sensor_id)) & (dataset.Data == str(chosen_date))]
dataset = dataset.drop(columns=["Zona","Contadores","ID_Espira","unique_id"])
tomtom = tomtom.loc[(tomtom.date == "10-4-2019") & (tomtom.id_location == 3)]
dataset = dataset.values
print(tomtom.head())



