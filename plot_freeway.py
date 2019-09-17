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
import random
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
default_date = '01/07/2004'
dataset = pd.read_csv("/home/vasco/tese/freeway_data/freeway_data0.csv")
dataset['Data'] = pd.to_datetime(dataset.Data)
dataset['Time'] = dataset['Data'].dt.strftime('%H:%M')
dataset['Day'] = dataset['Data'].dt.strftime('%d-%m-%Y')
dataset['Month'] = dataset['Data'].dt.strftime('%m')
dataset['Dow'] = dataset['Data'].dt.dayofweek

#dataset=dataset[dataset['Month']=='12']
#print(len(dataset))
int_to_dow = {"0": "monday", "1":"tuesday","2":"wednesday","3":"thursday","4":"friday", "5": "saturday", "6":"sunday"}
list_of_days = dataset.Day.unique()
#print(len(list_of_days))
dow = dataset.Dow.unique()
#print(dow)
selected_date = random.choice(list_of_days)
print(dataset.head())

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

dts = [dt.strftime('%H:%M') for dt in 
       datetime_range(datetime(2016, 9, 1, 0), datetime(2016, 9, 1, 23, 59), 
       timedelta(minutes=15))]


#selected_data = dataset[dataset['Day'] == selected_date]
#print(selected_data)
def plot_multiple_days(dataset,list_of_days):
	for j in list_of_days:
		selected_data = dataset[dataset['Day'] == j]
		fig, ax = plt.subplots()
		plt.xticks(rotation=90)
		ax.plot(selected_data['Time'],selected_data['Count'])
		x_ticks= ax.get_xticks()
		new_xticks = []
		for i in range(len(x_ticks)):
			if i%4 ==0:
				new_xticks.append(x_ticks[i])
		ax.set_xticks(new_xticks)
		ax.set_xlabel('Time of day')
		fig.savefig("/home/vasco/Desktop/freeway_plots/"+j+".png",dpi=100)
		plt.close()
def calc_avg_per_dow(dataset,dow):
	avg_per_d = {}
	avg_bdays = {}
	avg_wknds = {}
	for d in dow:
		avg_per_dow = {}
		selected_data = dataset[dataset['Dow'] == d]
		#print(d)
		data_np = selected_data.values
		for row in data_np:
			if row[2] not in avg_per_dow:
				avg_per_dow[row[2]] = [1,float(row[1])]
			else:
				new_value = avg_per_dow[row[2]][1] + float(row[1])
				new_n = avg_per_dow[row[2]][0] + 1
				avg_per_dow[row[2]] = [new_n,new_value]
		#print(avg_per_dow)
		average_count= []
		for k in avg_per_dow.keys():
			count = avg_per_dow[k][1]/avg_per_dow[k][0]
			average_count.append(count)
		#print(len(average_count))
		avg_per_d[d] = average_count
	return avg_per_d
	#print(len(avg_per_d))

def avg_wknds(dataset):
	avg_wknds={}
	avg_bdays={}
	for d in dow:
		selected_data = dataset[dataset['Dow'] == d]
		print(d)
		data_np = selected_data.values
		for row in data_np:
			if (d == 5) or (d == 6):
				if row[2] not in avg_wknds:
					avg_wknds[row[2]] = [1,float(row[1])]
				else:
					new_value = avg_wknds[row[2]][1] + float(row[1])
					new_n = avg_wknds[row[2]][0] + 1
					avg_wknds[row[2]] = [new_n,new_value]
			else:
				if row[2] not in avg_bdays:
					avg_bdays[row[2]] = [1,float(row[1])]
				else:
					new_value = avg_bdays[row[2]][1] + float(row[1])
					new_n = avg_bdays[row[2]][0] + 1
					avg_bdays[row[2]] = [new_n,new_value]
	avg_wknds_lst=[]
	avg_bdays_lst=[]
	for k in avg_wknds.keys():
		count = avg_bdays[k][1]/avg_bdays[k][0]
		avg_bdays_lst.append(count)
		count = avg_wknds[k][1]/avg_wknds[k][0]
		avg_wknds_lst.append(count)
		#for k in avg_per_dow.keys():
		#	count = avg_per_dow[k][1]/avg_per_dow[k][0]
		#	average_count.append(count)
			#print(len(average_count))

	#return avg_per_d
	return avg_bdays_lst,avg_wknds_lst
def plot_day(chosen_avg,dts,chosen_day):
	fig, ax = plt.subplots()
	plt.xticks(rotation=90)
	ax.plot(dts,chosen_avg)
	x_ticks= ax.get_xticks()
	new_xticks = []
	for i in range(len(x_ticks)):
		if i%4 ==0:
			new_xticks.append(x_ticks[i])
	ax.set_xticks(new_xticks)
	ax.set_xlabel('Time of day', fontsize=18)
	day = int_to_dow[str(chosen_day)]
	ax.set_title(day + " avg flow",fontsize=18)
	ax.set_ylabel('Traffic flow',fontsize=16)
	#plt.subplots_adjust(top=0.935,bottom=0.145,left=0.065,right=0.989,hspace=0.2,wspace=0.2)
	#plt.tight_layout()
	#plt.subplots_adjust(bottom=0.19)
	plt.show()
	plt.close()
	fig.savefig("/home/vasco/Desktop/freeway_plots/"+day+"_avg.png",dpi=100)
def plot_comparison(dts,avg_bdays_lst,avg_wknds_lst):
	fig, ax = plt.subplots(nrows=2,sharex=True)
	plt.xticks(rotation=90)
	ax[0].plot(dts,avg_bdays_lst)
	x_ticks= ax[0].get_xticks()
	new_xticks = []
	for i in range(len(x_ticks)):
		if i%4 ==0:
			new_xticks.append(x_ticks[i])
	ax[0].set_xticks(new_xticks)
	#ax[0].set_xlabel('Time of day', fontsize=14)
	ax[0].set_title("business day avg flow",fontsize=14)
	ax[0].set_ylabel('Traffic flow',fontsize=14)

	ax[1].plot(dts,avg_wknds_lst)
	x_ticks= ax[1].get_xticks()
	new_xticks = []
	for i in range(len(x_ticks)):
		if i%4 ==0:
			new_xticks.append(x_ticks[i])
	ax[1].set_xticks(new_xticks)
	ax[0].tick_params(axis='x', labelrotation=90)
	ax[1].set_xlabel('Time of day', fontsize=14)
	ax[1].set_title("weekend avg flow",fontsize=14)
	ax[1].set_ylabel('Traffic flow',fontsize=14)
	ax[0].grid(True)
	ax[1].grid(True)
	fig.savefig("/home/vasco/Desktop/freeway_plots/"+"comparison_bday_wknd.png",dpi=100)
	#plt.subplots_adjust(top=0.935,bottom=0.145,left=0.065,right=0.989,hspace=0.2,wspace=0.2)
	#plt.tight_layout()
	#plt.subplots_adjust(bottom=0.19)
	plt.show()

	plt.close()
	
avg_per_day = calc_avg_per_dow(dataset,dow)

chosen_day = input("Choose day of week 0 = Monday; 6 = Sunday: ")

if chosen_day:
	chosen_avg = avg_per_day[int(chosen_day)]
	plot_day(chosen_avg,dts,chosen_day)

avg_bdays_lst,avg_wknds_lst = avg_wknds(dataset)
plot_comparison(dts,avg_bdays_lst,avg_wknds_lst)
#print(dataset.head())
data_np = dataset.values
#print(data_np)
