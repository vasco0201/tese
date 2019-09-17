import copy
import os
from os import listdir
from os.path import isfile, join
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import sys
import numpy as np
import csv


espiras_cont = {} #nr de registos de cada espira, soma das contagens por hora para calcular media e mediana
espiras_cont_old = {}
n_errors = {} #contar os valores que estão acima do threshold por hora FIXME
espiras_geo = []
zonas =[]
avg_total = []
zone = input("Select the desired zone: ")
ID_Espira = input("Select the desired sensor ID): ")
date = input("Input the desired date (yyyy-mm-dd):")
contagens = open("/home/vasco/tese/dados_camara_todos.csv", "r") #dados 2018
reader = csv.reader(contagens)
date_dt = np.datetime64(date)
busday = date_dt.astype(datetime).isoweekday()
next(reader, None) #skips the header
i = 0
q = 0
desired_flow = []
for row in reader:
	temp_date = np.datetime64(row[0])
	temp_bday = temp_date.astype(datetime).isoweekday()
	if row[1] == str(zone) and row[3][2:] == str(ID_Espira) and temp_bday == busday:
		if q==0:
			print(temp_bday)
			print(temp_date)
			q=1

		if row[3][2:] == "":
			continue
		if (row[0] == date):
			desired_flow = row[4:]
			continue
		if (row[1],row[3][2:]) not in espiras_cont: #o row[3][2:] e para ignorar o ct e manter so o id da espira
		    espiras_cont[(row[1], row[3][2:])] = row[4:]
		    espiras_cont[(row[1], row[3][2:])].insert(0,1)
		else:
		    espiras_cont[(row[1],row[3][2:])][1:] = [float(x) + float(y) for x, y in zip(espiras_cont[(row[1],row[3][2:])][1:],row[4:])]
		    espiras_cont[(row[1],row[3][2:])][0] += 1
contagens.close()

#first I should get the average flow per hour for this one
avg_cont = copy.deepcopy(espiras_cont)
#print(avg_cont)
for k in avg_cont:
    for i in range(len(avg_cont[k])):
        if i > 0:
            avg_cont[k][i] = float(avg_cont[k][i])/float(avg_cont[k][0])

av_flow = avg_cont[(zone,ID_Espira)][1:]
perc_increase = []
for i in range(len(desired_flow)):
	#print i
	desired_flow[i] = float(desired_flow[i])
	perc_increase.append(av_flow[1]*100/desired_flow[i])

#print(av_flow)
print("AVG avg: ", av_flow[20])
print("AVG target: ", desired_flow[20])
print("perc", perc_increase[20])

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

dts = [dt.strftime('%H:%M') for dt in 
       datetime_range(datetime(2016, 9, 1, 0), datetime(2016, 9, 1, 23, 59), 
       timedelta(minutes=15))]
#print len(dts)
def plot_2scale(dts, desired_flow,av_flow,date):
	fig, ax = plt.subplots(nrows=1, sharex=True)
	#plt.locator_params(nbins=15, axis='x')
	plt.xticks(rotation=90)
	color = 'tab:blue'
	lbls=["Flow at " + str(date),"Avg flow"]#, "increase (\%)"]
	ax.set_ylabel('Traffic flow at '+ str(date),color=color,fontsize=22)
	l1 = ax.plot(dts,desired_flow, label="Flow at " + str(date),color=color,linewidth=2.0)
	ax.tick_params(axis='y', labelcolor=color,labelsize=20)
	ax.tick_params(axis='x', labelcolor='black',labelsize=14)
	#ax[0].legend(loc=1, prop={'size': 10})
	ax.grid(True)
	ax.set_title("Malfunctioning sensor example",fontsize=24)
	#fig.set_size_inches(15, 20)
	ax2 = ax.twinx()
	color = 'tab:red'
	ax2.set_ylabel('Avg traffic flow',color=color,fontsize=22)
	l2= ax2.plot(dts,av_flow, label="Avg flow of same business day",color=color,linewidth=2.0)
	#ax2.legend(loc=1, prop={'size': 10})
	ax2.tick_params(axis='y', labelcolor=color,labelsize=20)
	#ax[0].legend(loc=1, prop={'size': 10})
	#l3 = ax[1].plot(dts,perc_increase,color="darkgreen",linewidth=2.0)
	x_ticks= ax.get_xticks()
	new_xticks = []
	for i in range(len(x_ticks)):
		if i%4 ==0:
			new_xticks.append(x_ticks[i])
	ax.set_xticks(new_xticks)
	ax.set_xlabel('Time of day')
	#art=[]
	#lgd= ax.legend([l1,l2],labels=lbls,loc=9,bbox_to_anchor=[0.5, -0.1], ncol=2)
	#art.append(lgd)
	#plt.subplots_adjust(right=0.85)
	# here you can format your datetick labels as desired
	
	fig.savefig(os.getcwd() + "/example_malfunction.png", dpi=100)
	plt.show()
plot_2scale(dts, desired_flow,av_flow,date)
