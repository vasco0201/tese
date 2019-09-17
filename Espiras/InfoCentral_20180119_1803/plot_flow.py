import os
from os import listdir
from os.path import isfile, join
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv("/home/vasco/tese/dados_camara_todos.csv")#dados mais recentes
espira_ID = input("Select the desired sensor (ZONE_ctID) :")

dataset['unique_id'] = dataset.Zona.astype(str) + '_' + dataset.ID_Espira.astype(str)
dataset['unique_id'] = dataset['unique_id'].str.lower()
dataset = dataset[dataset["unique_id"] == str(ID_Espira)]

date = input("Input the desired date (yyyy-mm-dd):")
dataset =  dataset[dataset["Date"] == str(date)]
print(dataset.head())
sys.exit()

flow_data = []
lines = f.readlines()
for l in lines:
	data = l.split("|")
	data.remove("")
	data = data[:-1]
	if data[3] == "CT18" or data[3] == "CT5":
		print(len(data))
		print(data)
		temp = [data[0],data[3]]
		for i in data[4:]:
			#print i
			i = int(i)
			temp.insert(len(temp),i)
		flow_data.insert(len(flow_data), temp)

print(flow_data[0][1])
def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

dts = [dt.strftime('%H:%M') for dt in 
       datetime_range(datetime(2016, 9, 1, 0), datetime(2016, 9, 1, 23, 59), 
       timedelta(minutes=15))]
#print len(dts)

fig = plt.figure()
plt.xticks(rotation=90)
ax = fig.add_subplot(111)
ax.plot(dts,flow_data[0][2:], label=flow_data[0][1])
ax.plot(dts,flow_data[1][2:], label=flow_data[1][1])
ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,prop={'size': 15},title="ID Espira")
ax.grid(True)
ax.set_title("Cruzamento 98 "+ flow_data[0][0])
fig.set_size_inches(18.5, 12.5)
fig.savefig(os.getcwd() +"\\" + flow_data[0][0] + "_zona7.png", dpi=100)