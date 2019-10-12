import pandas as pd
import numpy as np
import copy
import sys
import os
import time
import matplotlib.pyplot as plt
data = pd.read_csv("train_mapets.csv")
print(data.columns)
train_lst= []
for col in data.columns: 
    train_lst.append(data[col].mean())

test_lst=[]

data = pd.read_csv("test_mapets.csv")
for col in data.columns: 
    test_lst.append(data[col].mean())

print(train_lst)
print("\n\n")
print(test_lst)

def plot_mape(train_mape_lst,test_mape_lst,flag_agg=0):
	fig, ax = plt.subplots()
	if flag_agg==0:
		x = range(1,len(train_mape_lst)+1)
		ax.set_title("Variation of mape with time window increase",fontsize=14)
		ax.set_ylabel('Mape',fontsize=14)
		ax.set_xlabel('Time window size', fontsize=14)
		title = "mape_timewindow.png"
	else:
		x = [15,30,45,60]
		ax.set_title("Variation of mape with increase in prediction horizon",fontsize=14)
		ax.set_ylabel('Mape',fontsize=14)
		ax.set_xlabel('Prediction horizon', fontsize=14)
		title = "mape_aggregation.png"
	ax.plot(x,train_mape_lst,label="Train",color="red",linewidth=2.0)
	ax.plot(x,test_mape_lst,label="Test",color="blue",linewidth=2.0)
	ax.legend(loc='best')
	ax.set_xticks(x)
	#ax[0].set_xlabel('Time of day', fontsize=14)
	ax.grid(True)
	fig.savefig("/home/vasco/Desktop/freeway_plots/"+str(title),dpi=100)
	#plt.subplots_adjust(top=0.935,bottom=0.145,left=0.065,right=0.989,hspace=0.2,wspace=0.2)
	#plt.tight_layout()
	#plt.subplots_adjust(bottom=0.19)
	plt.show()

	plt.close()

plot_mape(train_lst,test_lst,0)

