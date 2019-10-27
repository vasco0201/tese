import pandas as pd
import numpy as np
import copy
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import datetime
from scipy import stats
import statistics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import regularizers
import math
import csv
from os import listdir
from os.path import isfile, join
import os.path
import pickle
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import gc
from sklearn.metrics import mean_squared_error

from math import sqrt

curr_dir = os.getcwd()

def create_dataset(data):
    X,Y = [],[]
    for i in data:
        X.append(i[:-1])
        Y.append(i[-1])
    return np.array(X), np.array(Y)

def get_data(ID_Espira):
	data_folder = os.path.join(curr_dir,"dados_camara_todos.csv")
	dataset = pd.read_csv(data_folder)
	#dataset = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\CT15Mn-150818_101018\\dados_camara.csv")#dados mais recentes
	#tomtom = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\tomtom_data.csv")
	#dataset = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\periodic_data.csv") #dados periodicos gerados automaticamente
	#dataset = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\dados_old.csv") #dados mais antigos
	dataset['unique_id'] = dataset.Zona.astype(str) + '_' + dataset.ID_Espira.astype(str)
	dataset['unique_id'] = dataset['unique_id'].str.lower()
	dataset_uid = dataset[(dataset["unique_id"] == str(ID_Espira))]

	dataset_uid = dataset_uid.drop(columns=["Zona","Contadores","ID_Espira","unique_id"])
	dt2 = copy.deepcopy(dataset_uid)
	#dataset.sort_values(['Data'],ascending=True).groupby('Data').reset_index()

	dataset_uid = dataset_uid.groupby('Data').apply(lambda x: x.reset_index())

	# Split the dataset randomly
	msk = np.random.rand(len(dataset_uid)) < 0.8
	train_df = dataset_uid[msk]
	test_df = dataset_uid[~msk]
	

	## Split the dataset keeping in mind the sequence

	#dataset_2 = dataset_uid.values
	#train_size = int(len(dataset_2) * 0.80)
	#test_size = len(dataset_2) - train_size
	#train, test = dataset_2[0:train_size,:], dataset_2[train_size:len(dataset_2),:]
	
	#train_df = pd.DataFrame(train)
	#test_df = pd.DataFrame(test)

	#train_df = train_df.drop(columns=[0])
	#test_df = test_df.drop(columns=[0])
	dt2.to_csv("lstm/limited_data.csv", sep= ',', index=False)
	#train_set
	#train_df = train_df.drop(columns=["index"])

	train_df.to_csv("lstm/train.csv", sep= ',', index=False)

	#test_set
	#test_df = test_df.drop(columns=["index"])
	test_df.to_csv("lstm/test.csv", sep= ',', index=False)

	#full set
	dataset_uid = dataset_uid.drop(columns=["index"])
	dataset_uid.to_csv("lstm/dados_nn.csv", sep=',', index=False)
	return train_df, test_df


def smooth_data_old(train_cp,filename):
    data = train_cp.iloc[:,0]
    print(len(data.values))
    train_cp = train_cp.drop(columns=[1])
    train_cp = train_cp.values

    #train_cp = train_cp.astype('float32')

    avg_list = []
    std_list = []
    for i in range(len(train_cp[0])):
        curr_avg = sum(train_cp[:,i])/len(train_cp[:,i])
        avg_list.append(curr_avg)
        std_list.append(statistics.stdev(train_cp[:,i]))
    #FIX sera que e preciso guardar os valores antigos e usá-los para fazer os updates? 
    #agora esta a fazer uma especie de moving average, utilizando os valores novos nos updates seguintes


    number_of_changes= 0
    for row in range(len(train_cp)):
        
        for column in range(len(train_cp[0])):
            if (train_cp[row][column] > (avg_list[column] + 2*std_list[column])):
                old_value = train_cp[row][column]
                if column == 0:
                    previous_t = train_cp[row,-1]
                    next_t = train_cp[row,column+1]
                    train_cp[row,column] = (previous_t + next_t)/2
                    number_of_changes+=1
                if column == 95:
                    previous_t = train_cp[row,column-1]
                    next_t = train_cp[row,0]
                    train_cp[row,column] = (previous_t + next_t)/2
                    number_of_changes+=1
                else:
                    previous_t = train_cp[row,column-1]
                    next_t = train_cp[row,column+1]
                    train_cp[row,column] = (previous_t + next_t)/2
                    number_of_changes+=1
                new_value = train_cp[row][column]
    print("Number of changes: " + str(number_of_changes))
                
    new_d = data.values.reshape(len(data.values),1)
    test = np.concatenate((new_d,train_cp),axis=1)
    pd.DataFrame(test).to_csv("lstm/" + str(filename) + ".csv", sep=',', index=False)
    return test, train_cp

def smooth_data(train_cp,filename):
    data = train_cp.iloc[:,0]
    data_np = data.values
    #print(data.values)
    train_cp = train_cp.drop(columns=[1])
    train_cp = train_cp.values
    #train_cp = train_cp.astype('float32')
    avg_list = []
    std_list = []
    for i in range(len(train_cp[0])):
        curr_avg = sum(train_cp[:,i])/len(train_cp[:,i])
        avg_list.append(curr_avg)
        std_list.append(statistics.stdev(train_cp[:,i]))
    #FIX sera que e preciso guardar os valores antigos e usá-los para fazer os updates? 
    #agora esta a fazer uma especie de moving average, utilizando os valores novos nos updates seguintes
    #se tiver mais de 5 zeros seguidos como e que se faz
    inicial_len= len(train_cp)
    to_delete = []
    consec_zeros=0
    number_of_changes= 0
    for row in range(len(train_cp)):
    	n_zeros = np.count_nonzero(train_cp[row]==0.0)
    	if n_zeros > 60:
    		to_delete.append(row)
    	for column in range(len(train_cp[0])):
    		if (train_cp[row][column] > (avg_list[column] + 2*std_list[column])):
    			old_value = train_cp[row][column]
    			if column == 0:
    				previous_t = train_cp[row,-1]
    				next_t = train_cp[row,column+1]
    				train_cp[row][column] = (previous_t + train_cp[row][column] + next_t)/3
    				number_of_changes+=1
    			if column == 95:
    				previous_t = train_cp[row,column-1]
    				next_t = train_cp[row,0]
    				train_cp[row][column] = (previous_t + train_cp[row][column] + next_t)/3
    				number_of_changes+=1
    			else:
    				previous_t = train_cp[row,column-1]
    				next_t = train_cp[row,column+1]
    				train_cp[row][column] = (previous_t + train_cp[row][column] + next_t)/3
    				number_of_changes+=1
    			new_value = train_cp[row][column]
    			#	train_cp[row][column] = avg_list[column]
    	#if n_zeros >= int(0.25*len(train_cp[0])):
        	#delete this entry
        #	to_delete.append(row)
    #x = copy.deepcopy(train_cp)
    
    train_cp = np.delete(train_cp, to_delete, axis=0)
    data_np = np.delete(data_np,to_delete,axis=0)
    #print("Number of changes: " + str(number_of_changes))
    #print("Number of times that there are 3 consecutive zeros:", consec_zeros)        
    print("Numero de dias inuteis / total dias:", len(to_delete),"/",inicial_len)
    new_d = data_np.reshape(len(data_np),1)
    test = np.concatenate((new_d,train_cp),axis=1)
    pd.DataFrame(test).to_csv("lstm/" + str(filename) + ".csv", sep=',', index=False)
    return test, train_cp

def transform_data(in_file, out_file, nrows=-1):
    in_file = open(str(in_file),"r")
    next(in_file)
    out_file = open(str(out_file),"w")
    out_file.write("t-3,t-2,t-1,Y\n") #header
    k = 0    
    lines = in_file.readlines()
    for line in lines:
        line = line.split(",")
        line = line[1:]
        line[-1] = line[-1].replace("\n","") #last data record has a \n
        its = [iter(line), iter(line[1:]), iter(line[2:]),iter(line[3:])] #Construct the pattern for longer windowss
        x = list(zip(*its))
        if (k == nrows):
            break
            
        k+=1 
	#print(x)
    #j = 0
    #while(j<50): #this cycle was for creating a mock dataset with repeated data for testing purposes
        for i in x:
        #print(i[0],i[1],i[2],i[3])
            out_file.write(i[0] + "," + i[1] +"," + i[2] +","+ i[3])
            out_file.write("\n")
     #   j+=1
    
    in_file.close()
    out_file.close()

def transform_data2(in_file, out_file, time_lags=4):
    in_file = open(str(in_file),"r")
    next(in_file)
    out_file = open(str(out_file),"w")
    n=1
    for lag in range(time_lags):
    	if lag == time_lags-1:
    		out_file.write("Y\n")
    	else:
    		out_file.write("t-"+str(time_lags-n)+",")
    	n+=1
    #out_file.write("t-3,t-2,t-1,Y\n") #header
    k = 0    
    lines = in_file.readlines()
    for line in lines:
        line = line.split(",")
        line = line[1:]
        line[-1] = line[-1].replace("\n","") #last data record has a \n
        its = []
        for lag in range(time_lags):
        	if lag ==0:
        		its = [iter(line)]
        	else:
        		its.append(iter(line[lag:]))

       	#its = [iter(line), iter(line[1:]), iter(line[2:]),iter(line[3:])] #Construct the pattern for longer windowss
        x = list(zip(*its))
        k+=1

        for i in x:
        	for u in range(len(i)):
        		if u == len(i)-1:
        			out_file.write(i[u]+"\n")
        		else:
        			out_file.write(i[u]+",")
    
    in_file.close()
    out_file.close()

def aggregate_data(dataset,filename, interval):
	new_data = []
	for i in range(len(dataset)):
		j = 1
		temp=[]
		if (int(interval) == 15):
			
			pd.DataFrame(dataset).to_csv("lstm/"+str(filename) + "_15.csv", sep=',', index=False)
			return dataset

		while (j < len(dataset[i][1:])):
			if int(interval) == 30:
				value = dataset[i][j]+dataset[i][j+1]
				temp.append(value)
				j+=2
			elif int(interval) == 45:
				value = dataset[i][j]+dataset[i][j+1]+ dataset[i][j+2]
				temp.append(value)
				j+=3
			elif int(interval) == 60:
				value = dataset[i][j]+dataset[i][j+1]+ dataset[i][j+2] + dataset[i][j+3]
				temp.append(value)
				j+=4
			else: 
				print("Invalid interval, using default : 15 min")
				pd.DataFrame(dataset).to_csv("lstm/"+str(filename) + "_15.csv", sep=',', index=False)
				return dataset
		temp.insert(0,dataset[i][0])
		new_data.append(temp)
	new_data = np.asarray(new_data)
	pd.DataFrame(new_data).to_csv("lstm/" + str(filename) + "_" + str(interval) + ".csv", sep=',', index=False)
def get_nn_data(filename):
	dataframe = pd.read_csv(str(filename))
	dataset = dataframe.values
	dataset = dataset.astype('float32')
	x, y = create_dataset(dataset)
	return	x, y

def lstm_model(trainX,trainY,n_features,n_steps,n_epochs):
	trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], n_features))
	adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model = Sequential()
	model.add(LSTM(16, activation='relu', input_shape=(n_steps, n_features), return_sequences=False ,kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01)))
	#model.add(LSTM(64, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01)))
	
	model.add(Dense(1))
	model.compile(optimizer=adam, loss='mse',metrics=['mse','mae','mape'])
	# fit model
	patience= int(round(n_epochs/3))
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

	history = model.fit(trainX, trainY, epochs=n_epochs, verbose=0,validation_split=0.20, callbacks=[es])
	return model,history

def model_results(trainX, trainY, testX, testY,model):
	# Estimate model performance
	trainScore = model.evaluate(trainX, trainY, verbose=0)
	testScore = model.evaluate(testX, testY, verbose=0)
	return trainScore, testScore

def evaluate_instance(filename, model,n_features):
	obs_df = pd.read_csv(filename)
	obs = obs_df.values
	obs = obs.astype('float32')
	obsX, obsY = create_dataset(obs)
	obsX = obsX.reshape((obsX.shape[0], obsX.shape[1], n_features))
	#pred = model.predict(obsX)
	score = model.evaluate(obsX, obsY, verbose=0)
	print(model.metrics_names)
	print("MAE:" ,score[2] ,"MSE: ", score[1])
def evaluate_model(filename, model,n_features,flag=0, obsY="data"):
	filename = filename.reshape((filename.shape[0], filename.shape[1], n_features))
	to_delete=[]
	for i in range(len(obsY)):
		if obsY[i] == 0:
			to_delete.append(i)

	obs_withoutZero = np.delete(obsY,to_delete, axis=0)
	filename_without_zero = np.delete(filename,to_delete,axis=0)

	obs_onlyZero = obsY[to_delete]
	filename_only_zero = filename[to_delete]
	#print("MAPE: ", m_mape)

	score = model.evaluate(filename, obsY, verbose=0)
	rmse_total = sqrt(score[1])
	if len(filename_only_zero) != 0:
		score_no_zeros = model.evaluate(filename_without_zero, obs_withoutZero, verbose=0)
		score_of_zeros = model.evaluate(filename_only_zero, obs_onlyZero, verbose=0)
		return score_no_zeros[3], score_no_zeros[2], sqrt(score_no_zeros[1]), score_of_zeros[2], sqrt(score_of_zeros[1]), score[0], rmse_total, score[2]
	else:
		return score[3], 1, 1, 1, 1, score[0], rmse_total, score[2]
	#print("RMSE:",sqrt(scoreWithoutzeros[1]), "MAE:", scoreWithoutzeros[2], "MAPE:", scoreWithoutzeros[3])

	#print("MAE:" ,score[2] ,"MSE: ", score[1])
# def mean_absolute_percentage_error(y_true, y_pred): 
# 	y_true, y_pred = np.array(y_true), np.array(y_pred)
# 	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
	
def calc_mape(pred,obsY):
	mape = []
	for i in range(len(pred)):
		err = abs((pred[i]+1)-(obsY[i]+1))/abs(obsY[i]+1)
		if err > 1000:
			print(err)
			print("loool")
			break
		mape.append(err)
	return sum(mape)*100/(len(mape))
################## FREEWAY DATA ########################
def freeway_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    #print(dataset)
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)



def freeway_preprocess(filename,interval=15):
	dataframe = pd.read_csv(str(filename), usecols=[1], engine='python')
	dataset = dataframe.values
	dataset = dataset.astype('float32')
	new_dataset=[]
	i = 0
	while i < len(dataset):
		if interval == 30:
			temp = dataset[i]+dataset[i+1]
			i+=2
		elif interval == 45:
			temp = dataset[i]+dataset[i+1]+dataset[i+2]
			i+=3
		elif interval == 60:
			temp = dataset[i]+dataset[i+1]+dataset[i+2]+dataset[i+3]
			i+=4
		else: 
			new_dataset = dataset
			break
		new_dataset.append(temp)
	new_dataset = np.asarray(new_dataset)

	# split into train and test sets
	train_size = int(len(new_dataset) * 0.80)
	test_size = len(new_dataset) - train_size
	train, test = new_dataset[0:train_size,:], new_dataset[train_size:len(new_dataset),:]
	return train,test
def last_business_day(dates,trainX,trainY):
	new_x = []
	new_y = []
	#print(dates)
	first_week = 0
	length= len(trainX)
	for i in range(len(trainX)):
		curr_date = datetime.datetime.strptime(trainX[i][-1][0], '%Y-%m-%d %H:%M:%S')
		#print(curr_date.weekday())
		last_busday = curr_date - datetime.timedelta(days=7)
		temp_busday = str(last_busday)
		if ([temp_busday]) in dates:
			target = find_element(trainX, temp_busday)
			temp = copy.deepcopy(trainX[i][:,1])
			temp = np.concatenate((temp,[target[1]]),axis=0)
			
			
			new_x.append(temp) #ja tenho a cena que necessito
			new_y.append(trainY[i][1])
			#print(new_x)
			#sys.exit()
		#print(curr_date.weekday())
		#print(last_busday.weekday())
		#print(last_busday)
		if i%1000==0:
			print("Tou na iteracao ", i, " /",length)
		if i==1000:
			print(new_x)
			print(new_y)
	return np.array(new_x),np.array(new_y)
	#tenho de ver se existe nos dados odia anterior e se nao houver nao ponho no novo df
def split_dataset(dataset,train_ratio):
	train_size = int(len(dataset) * train_ratio)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	return train, test
def find_element(x,target):
	for i in range(len(x)):
		for j in range(len(x[i])):
			#print(x[i])
			if x[i][j][0] == target:
				return x[i][j]

def plot_error(train_mape_lst,test_mape_lst,flag_agg="input_size"):
	fig, ax = plt.subplots()
	if flag_agg=="input_size":
		x = range(1,len(train_mape_lst)+1)
		ax.set_title("Variation of mape with time window increase",fontsize=14)
		ax.set_ylabel('Mape',fontsize=14)
		ax.set_xlabel('Time window size', fontsize=14)
		title = "mape_timewindow.png"
	elif flag_agg=="pred_horizon":
		x = [15,30,45,60]
		ax.set_title("Variation of mape with increase in prediction horizon",fontsize=14)
		ax.set_ylabel('Mape',fontsize=14)
		ax.set_xlabel('Prediction horizon', fontsize=14)
		title = "mape_aggregation.png"
	elif flag_agg=="loss_nlayers":
		x = range(1,len(train_mape_lst)+1)
		ax.set_title("Variation of loss function",fontsize=14)
		ax.set_ylabel('Loss',fontsize=14)
		ax.set_xlabel('Number of layers', fontsize=14)
		title = "nlayers_loss.png"

	ax.plot(x,train_mape_lst,label="Train",color="red",linewidth=2.0)
	ax.plot(x,test_mape_lst,label="Test",color="blue",linewidth=2.0)
	ax.legend(loc='best')
	ax.set_xticks(x)
	#ax[0].set_xlabel('Time of day', fontsize=14)
	ax.grid(True)
	fig.savefig("/home/vasco/Desktop/freeway_plots/lstm_"+str(title),dpi=100)
	#plt.subplots_adjust(top=0.935,bottom=0.145,left=0.065,right=0.989,hspace=0.2,wspace=0.2)
	#plt.tight_layout()
	#plt.subplots_adjust(bottom=0.19)
	plt.show()

	plt.close()




#################### TESTS ####################
def mape_vs_timelag(filename,n_features,epochs):
	with open("lstm_train_mape_ts.txt", "rb") as fp:
		train_mape_ts = pickle.load(fp)
	with open("lstm_test_mape_ts.txt", "rb") as fp:
		test_mape_ts = pickle.load(fp)
	train, test = freeway_preprocess(filename,15)
	i=8
	for interval in [9,10]:
		print("------------------------------------------------------------------")
		print("Tou no:", interval)
		print("------------------------------------------------------------------")
		
		trainX, trainY = freeway_dataset(train,interval)
		testX, testY =freeway_dataset(test,interval)
		
		model, history = lstm_model(trainX,trainY,n_features,interval,epochs)
		
		mae,rmse,mape,mape_approx= evaluate_model(trainX,model,n_features,1,trainY)
		train_mape_ts[i].append(mape_approx)
		tmae,trmse,tmape,tmape_approx = evaluate_model(testX,model,n_features,1,testY)
		test_mape_ts[i].append(tmape_approx)
		i+=1
	
	for i in test_mape_ts:
		print(len(i))
	with open("lstm_train_mape_ts.txt", "wb") as fp:
 		pickle.dump(train_mape_ts, fp)
	with open("lstm_test_mape_ts.txt", "wb") as fp:
		pickle.dump(test_mape_ts, fp)

def include_business_day(filename):
	#loading the dataset
	dataframe = pd.read_csv(filename, engine='python')
	dataset = dataframe.values
	train,test = split_dataset(dataset,0.80)
	
	#loading all the dates in the dataset 
	dates_df = pd.read_csv(filename, usecols=[0],engine='python')
	dates_array = dates_df.values
	dates_train, dates_test = split_dataset(dates_array,0.80)


	
	#searching for the right previous dates and including them in the data that is going to be fed to the net
	#takes forever, after first run save it and then load them for the next runs
	# f_trainX, f_trainY = freeway_dataset(train,3)
	# f_trainX, f_trainY = last_business_day(dates_train,f_trainX,f_trainY)

	# f_trainX = f_trainX.astype('float32')
	# f_trainY = f_trainY.astype('float32')

	# f_testX, f_testY = freeway_dataset(test,3)
	# f_testX, f_testY = last_business_day(dates_test,f_testX,f_testY)
	
	# f_testX = f_testX.astype('float32')
	# f_testY = f_testY.astype('float32')
	
	##################################################################
	#saves the dataset with the same bus day since it takes forever to compute
	
	# with open("f_trainX.txt", "wb") as fp:
	# 	pickle.dump(f_trainX, fp)
	# with open("f_testX.txt", "wb") as fp:
	# 	pickle.dump(f_testX, fp)
	# with open("f_trainY.txt", "wb") as fp:
	# 	pickle.dump(f_trainY, fp)
	# with open("f_testY.txt", "wb") as fp:
	# 	pickle.dump(f_testY, fp)
	##################################################################

	#loading the data previously obtained
	with open("f_trainX.txt", "rb") as fp:
		f_trainX = pickle.load(fp)
	with open("f_testX.txt", "rb") as fp:
		f_testX = pickle.load(fp)
	with open("f_trainY.txt", "rb") as fp:
		f_trainY = pickle.load(fp)
	with open("f_testY.txt", "rb") as fp:
		f_testY = pickle.load(fp)
	return f_trainX,f_trainY,f_testX,f_testY

def run_multi_aggregation(filename,epochs,n_steps,n_features):
	#f = open("lstm_mape_multi_aggregation.csv","a+")
	#f.write("15,30,45,60\n")
	train_mape_file = open("/content/drive/My Drive/freeway/lstm_train_mape_agg.txt", "wb")
	train_mape_file.close()
	
	#g = open("test_lstm_mape_multi_aggregation.csv","a+")
	#g.write("15,30,45,60\n")
	train_mape_agg = [[],[],[],[]]
	test_mape_agg = [[],[],[],[]]




	#train_rmse_zeros_agg = [[],[],[],[]]
	#test_rmse_zeros_agg = [[],[],[],[]]

	#train_mae_zeros_agg = [[],[],[],[]]
	#test_mae_zeros_agg = [[],[],[],[]]

	train_rmse_total_agg = [[],[],[],[]]
	test_rmse_total_agg = [[],[],[],[]]

	train_mae_total_agg = [[],[],[],[]]
	test_mae_total_agg = [[],[],[],[]]

	#h = open("lstm_mape_multi_aggregation.csv","a+")
	#h.write("15,30,45,60\n")
	for run in range(5):
		i=0
		print("Run nr:", run+1,"/5")
		for interval in [15,30,45,60]:
			print("Agg:", interval)
			train,test = freeway_preprocess(filename,interval)

			trainX, trainY = freeway_dataset(train,n_steps)
			testX, testY =freeway_dataset(test,n_steps)

			model, history = lstm_model(trainX,trainY,n_features,n_steps,epochs)

			mape, mae, rmse, mae_zeros, rmse_zeros, loss, rmse_total, mae_total  = evaluate_model(trainX, model, 1,1,trainY)
			tmape, tmae, trmse, tmae_zeros, trmse_zeros, tloss, trmse_total, tmae_total  = evaluate_model(testX, model, 1, 1,testY)

			train_mape_agg[i].append(mape)
			test_mape_agg[i].append(tmape)

			train_rmse_total_agg[i].append(rmse_total)
			test_rmse_total_agg[i].append(trmse_total)

			train_mae_total_agg[i].append(mae_total)
			test_mae_total_agg[i].append(tmae_total)

			
			i+=1


		train_mape_file = open("/content/drive/My Drive/freeway/lstm_train_mape_agg.txt", "wb")
		test_mape_file = open("/content/drive/My Drive/freeway/lstm_test_mape_agg.txt", "wb")

		train_rmse_total_file = open("/content/drive/My Drive/freeway/lstm_train_rmsetotal_agg.txt", "wb")
		test_rmse_total_file = open("/content/drive/My Drive/freeway/lstm_test_rmsetotal_agg.txt", "wb")

		train_mae_total_file = open("/content/drive/My Drive/freeway/lstm_train_maetotal_agg.txt", "wb")
		test_mae_total_file = open("/content/drive/My Drive/freeway/lstm_test_maetotal_agg.txt", "wb")

		pickle.dump(train_mape_agg,train_mape_file)
		pickle.dump(test_mape_agg,test_mape_file)
		pickle.dump(train_rmse_total_agg,train_rmse_total_file)
		pickle.dump(test_rmse_total_agg,test_rmse_total_file)
		pickle.dump(train_mae_total_agg,train_mae_total_file)
		pickle.dump(test_mae_total_agg,test_mae_total_file)

		train_mape_file.close()
		test_mape_file.close()
		train_rmse_total_file.close()
		test_rmse_total_file.close()
		train_mae_total_file.close()
		test_mae_total_file.close()



################################################################################



#################################### URBAN DATA ################################
def urban_mape_timelag(agg,epochs):
	train_mape_evo = [[],[],[],[],[],[],[],[],[],[]]
	test_mape_evo = [[],[],[],[],[],[],[],[],[],[]]

	train_rmse_evo = [[],[],[],[],[],[],[],[],[],[]]
	test_rmse_evo = [[],[],[],[],[],[],[],[],[],[]]
	train_mape_file = open("/content/drive/My Drive/4_ct6/ts_test/lstm_urban_train_mape_ts.txt", "wb")
	train_mape_file.close()

	for run in range(3):
		i = 0
		print("Run:", run+1,"/3")
		for lag in range(2,12):
			print("Lag:", lag-1, "/10")

			transform_data2("lstm/train_"+str(agg)+".csv","lstm/train_formatted.csv",lag)
			transform_data2("lstm/test_"+str(agg)+".csv","lstm/test_formatted.csv",lag)

			trainX, trainY = get_nn_data('lstm/train_formatted.csv')
			testX, testY = get_nn_data('lstm/test_formatted.csv')

			model, history= lstm_model(trainX,trainY,1,lag-1,epochs) #Eventualmente dar a opcao de escolher os hiperparametros

			mape, mae, rmse, mae_zeros, rmse_zeros, loss, rmse_total, mae_total  = evaluate_model(trainX, model, 1,1,trainY)
			tmape, tmae, trmse, tmae_zeros, trmse_zeros, tloss, trmse_total, tmae_total  = evaluate_model(testX, model, 1, 1,testY)
			train_mape_evo[i].append(mape)
			test_mape_evo[i].append(tmape)

			train_rmse_evo[i].append(rmse_total)
			test_rmse_evo[i].append(trmse_total)
			
			print("Train:")
			print("MAPE:", mape, "RMSE:", rmse_total, "MAE:", mae_total)

			print("Test:")
			print("MAPE:", tmape, "RMSE:", trmse_total, "MAE:", tmae_total)

			i+=1
		print("Saving progress...")
		train_mape_file = open("/content/drive/My Drive/4_ct6/ts_test/lstm_urban_train_mape_ts.txt", "wb")
		test_mape_file = open("/content/drive/My Drive/4_ct6/ts_test/lstm_urban_test_mape_ts.txt","wb")

		train_rmse_file = open("/content/drive/My Drive/4_ct6/ts_test/lstm_urban_train_rmse_ts.txt", "wb")
		test_rmse_file = open("/content/drive/My Drive/4_ct6/ts_test/lstm_urban_test_rmse_ts.txt", "wb")
		
		pickle.dump(train_mape_evo,train_mape_file)
		pickle.dump(test_mape_evo,test_mape_file)

		pickle.dump(train_rmse_evo,train_rmse_file)
		pickle.dump(test_rmse_evo,test_rmse_file)
		
		train_mape_file.close()
		test_mape_file.close()
		train_rmse_file.close()
		test_rmse_file.close()

		print("Saved!")

def urban_multi_agg(train,test,n_steps=5):
	#train_mape_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_train_mape_agg.txt", "wb")
	#train_mape_file.close()

	train_mape_agg = [[],[],[],[]]
	test_mape_agg = [[],[],[],[]]

	train_rmse_agg = [[],[],[],[]]
	test_rmse_agg = [[],[],[],[]]

	train_mae_agg = [[],[],[],[]]
	test_mae_agg = [[],[],[],[]]


	train_rmse_zeros_agg = [[],[],[],[]]
	test_rmse_zeros_agg = [[],[],[],[]]

	train_mae_zeros_agg = [[],[],[],[]]
	test_mae_zeros_agg = [[],[],[],[]]

	train_rmse_total_agg = [[],[],[],[]]
	test_rmse_total_agg = [[],[],[],[]]

	train_mae_total_agg = [[],[],[],[]]
	test_mae_total_agg = [[],[],[],[]]


	for run in range(1):
		print("Run nr:", run+1,"/3")
		i=0
		for interval in [15,30,45,60]:
			print("Agg:", interval)
			aggregate_data(train,"train", interval)
			aggregate_data(test,"test",interval)

			transform_data2("lstm/train_"+str(interval)+".csv","lstm/train_formatted.csv",n_steps)
			transform_data2("lstm/test_"+str(interval)+".csv","lstm/test_formatted.csv",n_steps)

			trainX, trainY = get_nn_data('lstm/train_formatted.csv')
			testX, testY = get_nn_data('lstm/test_formatted.csv')
			
			model, history= lstm_model(trainX,trainY,1,n_steps-1,50) #Eventualmente dar a opcao de escolher os hiperparametros
			print("TRAIN:")
			mape, mae, rmse, mae_zeros, rmse_zeros, loss, rmse_total, mae_total  = evaluate_model(trainX, model, 1,1,trainY)
			print("TEST:")
			tmape, tmae, trmse, tmae_zeros, trmse_zeros, tloss, trmse_total, tmae_total  = evaluate_model(testX, model, 1, 1,testY)
			
			### Metrics for obs != 0
			train_mape_agg[i].append(mape)
			test_mape_agg[i].append(tmape)

			train_rmse_agg[i].append(rmse)
			test_rmse_agg[i].append(trmse)

			train_mae_agg[i].append(mae)
			test_mae_agg[i].append(tmae)

			### Metrics for obs = 0
			train_rmse_zeros_agg[i].append(rmse_zeros)
			test_rmse_zeros_agg[i].append(trmse_zeros)

			train_mae_zeros_agg[i].append(mae_zeros)
			test_mae_zeros_agg[i].append(tmae_zeros)

			### Metrics for all obs
			train_rmse_total_agg[i].append(rmse_total)
			test_rmse_total_agg[i].append(trmse_total)

			train_mae_total_agg[i].append(mae_total)
			test_mae_total_agg[i].append(tmae_total)



			print("Train:")
			print("MAPE:", mape, "RMSE:", rmse_total, "MAE:", mae_total)
			print("OBS = 0","MAPE:", "-", "RMSE:", rmse_zeros, "MAE:", mae_zeros)
			print("OBS != 0","MAPE:", mape, "RMSE:", rmse, "MAE:", mae)
			
			print("Test:")
			print("MAPE:", tmape, "RMSE:", trmse_total, "MAE:", tmae_total)
			print("OBS = 0","MAPE:", "-", "RMSE:", trmse_zeros, "MAE:", tmae_zeros)
			print("OBS != 0","MAPE:", tmape, "RMSE:", trmse, "MAE:", tmae)
			i+=1
		print("Saving progress...")

		# train_mape_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_train_mape_agg.txt", "wb")
		# test_mape_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_test_mape_agg.txt", "wb")

		# train_rmse_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_train_rmse_agg.txt", "wb")
		# test_rmse_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_test_rmse_agg.txt", "wb")

		# train_mae_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_train_mae_agg.txt", "wb")
		# test_mae_file =	open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_test_mae_agg.txt", "wb")

		# train_rmse_zeros_file  = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_train_rmseWzeros_agg.txt", "wb")
		# test_rmse_zeros_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_test_rmseWzeros_agg.txt", "wb")
		
		# train_mae_zeros_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_train_maeWzeros_agg.txt", "wb")
		# test_mae_zeros_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_test_maeWzeros_agg.txt", "wb")
		
		# train_rmse_total_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_train_rmsetotal_agg.txt", "wb")
		# test_rmse_total_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_test_rmsetotal_agg.txt", "wb")

		# train_mae_total_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_train_maetotal_agg.txt", "wb")
		# test_mae_total_file = open("/content/drive/My Drive/4_ct6/agg_test/lstm_urban_test_maetotal_agg.txt", "wb")
		
		# pickle.dump(train_mape_agg,train_mape_file)
		# pickle.dump(test_mape_agg,test_mape_file)

		# pickle.dump(train_rmse_agg,train_rmse_file)
		# pickle.dump(test_rmse_agg,test_rmse_file)

		# pickle.dump(train_mae_agg,train_mae_file)
		# pickle.dump(test_mae_agg,test_mae_file)

		# pickle.dump(train_rmse_zeros_agg,train_rmse_zeros_file)
		# pickle.dump(test_rmse_zeros_agg,test_rmse_zeros_file)

		# pickle.dump(train_mae_zeros_agg,train_mae_zeros_file)
		# pickle.dump(test_mae_zeros_agg,test_mae_zeros_file)

		# pickle.dump(train_rmse_total_agg,train_rmse_total_file)
		# pickle.dump(test_rmse_total_agg,test_rmse_total_file)

		# pickle.dump(train_mae_total_agg,train_mae_total_file)
		# pickle.dump(test_mae_total_agg,test_mae_total_file)



		# train_mape_file.close()
		# test_mape_file.close()
		# train_rmse_file.close()
		# test_rmse_file.close()
		# train_mae_file.close()
		# test_mae_file.close()
		# train_rmse_zeros_file.close()
		# test_rmse_zeros_file.close()
		# train_mae_zeros_file.close()
		# test_mae_zeros_file.close()
		# train_rmse_total_file.close()
		# test_rmse_total_file.close()
		# train_mae_total_file.close()
		# test_mae_total_file.close()

		print("Saved!")






################################################################################




def run_lstm(f_trainX,f_trainY,n_features,n_steps,epochs):
	model, history = lstm_model(f_trainX,f_trainY,n_features,n_steps,epochs)
	n_layers  = len(model.layers)-1
	n_units = []
	for i in model.layers:
		if "lstm" in i.get_config()['name']:
			n_units.append(i.get_config()['units'])

	#evaluate_model("lstm/train_formatted.csv", model,n_features)
	#evaluate_model("lstm/test_formatted.csv", model,n_features)
	mae,rmse,mape,mape_approx= evaluate_model(f_trainX,model,n_features,1,f_trainY)
	tmae,trmse,tmape,tmape_approx = evaluate_model(f_testX,model,n_features,1,f_testY)
	if not (os.path.exists("lstm_freeway_results.csv")):
		out_file =  csv.writer(open("lstm_freeway_results.csv","a+", newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
		header =["filename","nr_past_ts","epochs","hidden_layers","n_units","mae","rmse","mape","test_mae", "test_rmse", "test_mape"]
		out_file.writerow(header)
	else:
		out_file =  csv.writer(open("lstm_freeway_results.csv","a+", newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
	out_file.writerow(["freeway_data2.csv w/bday", n_steps, epochs, n_layers, n_units, mae, rmse, mape, tmae, trmse, tmape])

def main():
	n_features = 1
	n_steps = 5
	epochs = 100
	
	#x = input("1 - Espiras; 2 - Autoestradas :")
	x=1
	if x and int(x)==1:
		n_steps=5
		ID_Espira = "4_ct6"
		train_set, test_set = get_data(ID_Espira)

		train_cp = copy.deepcopy(train_set)
		test_cp = copy.deepcopy(test_set)
		
		smooth_train, df_train= smooth_data(train_cp, "train_2")
		smooth_test, df_test = smooth_data(test_cp, "test_2")


		#transform_data("lstm/train_2.csv","lstm/train_formatted.csv")	
		#transform_data("lstm/test_2.csv","lstm/test_formatted.csv")
		interval=15
		aggregate_data(smooth_train,"train", interval)
		aggregate_data(smooth_test,"test",interval)

		#transform_data2("lstm/train_"+str(interval)+".csv","lstm/train_formatted.csv",n_steps+1)
		#transform_data2("lstm/test_"+str(interval)+".csv","lstm/test_formatted.csv",n_steps+1)

		#f_trainX, f_trainY = get_nn_data('lstm/train_formatted.csv')
		#f_testX, f_testY = get_nn_data('lstm/test_formatted.csv')


		#model, history = lstm_model(f_trainX,f_trainY,n_features,n_steps,epochs)

		##### TEST RUNS ######

		# time lag variation
		#urban_mape_timelag(15,40)
		# aggregation test(train, test, n_prev steps + 1)
		urban_multi_agg(smooth_train,smooth_test,5)

		# n_layers  = len(model.layers)-1
		# n_units = []
		# for i in model.layers:
		# 	if "lstm" in i.get_config()['name']:
		# 		n_units.append(i.get_config()['units'])

		# 	#evaluate_model("lstm/train_formatted.csv", model,n_features)
		# 	#evaluate_model("lstm/test_formatted.csv", model,n_features)
		# 	mae,rmse,mape,mape_approx= evaluate_model(f_trainX,model,n_features,1,f_trainY)
		# 	tmae,trmse,tmape,tmape_approx = evaluate_model(f_testX,model,n_features,1,f_testY)
		# 	if not (os.path.exists("lstm_loopsensor_results.csv")):
		# 		out_file =  csv.writer(open("lstm_urban_results.csv","a+", newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
		# 		header =["Espira","nr_past_ts","epochs","hidden_layers","n_units","mae","rmse","mape","test_mae", "test_rmse", "test_mape"]
		# 		out_file.writerow(header)
		# 	else:
		# 		out_file =  csv.writer(open("lstm_urban_results.csv","a+", newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
		# 	out_file.writerow([ID_Espira, n_steps, epochs, n_layers, n_units, mae, rmse, mape_approx, tmae, trmse, tmape_approx])

	else: ################## freeway dataset ##########################
		#### Include data from the previous day at the same time bus_day =1
		f_trainX, f_trainY = [],[]
		f_testX, f_testY = [],[]


		bus_day=0
		if bus_day ==0:

			######################################################################			
			#tests the network with multiple input sizes (number of prev occurrences that factor in the pred)
			#mape_vs_timelag("freeway_data/freeway_data2.csv",n_features,epochs)

			# with open("lstm_train_mape_ts.txt", "rb") as fp:
			# 	train_mape_ts = pickle.load(fp)
			# with open("lstm_test_mape_ts.txt", "rb") as fp:
			# 	test_mape_ts = pickle.load(fp)
			# train_mape_ts= train_mape_ts[:-2]
			# test_mape_ts = test_mape_ts[:-2]
			# avg_train_mapets = []
			# avg_test_mapets = []
			# print(train_mape_ts)
			# for i in range(len(train_mape_ts)):
			# 	avg_train_mapets.append((sum(train_mape_ts[i])/len(train_mape_ts[i]))[0])
			# 	avg_test_mapets.append((sum(test_mape_ts[i])/len(test_mape_ts[i]))[0])

			# plot_error(avg_train_mapets,avg_test_mapets,flag_agg="input_size")
			#tests performance for multiple time_steps
			print(n_steps)
			print(n_features)
			print(epochs)
			run_multi_aggregation("freeway_data/freeway_data2.csv",epochs,n_steps,n_features)
			######################################################################

			#preparation for a normal run of the network
			# dataframe = pd.read_csv('freeway_data/freeway_data2.csv',usecols=[1] ,engine='python')
			# dataset = dataframe.values
			# dataset = dataset.astype('float32')
			# train,test = split_dataset(dataset,0.80)
			# f_trainX, f_trainY = freeway_dataset(train,4)
			# f_testX, f_testY = freeway_dataset(test,4)
			# run_lstm(f_trainX,f_trainY,n_features,len(f_trainX[0]),epochs)
		



		elif bus_day ==1:
			# new_file = open("temp_file.txt","w")
			# train,test = split_dataset(dataset,0.80)
			# trainXold, trainYold = freeway_dataset(train,3)
			# for i in range(len(trainXold)):
			# 	new_file.write(str(trainXold[i][0][0]) + " " + str(trainXold[i][1][0]) + " " + str(trainXold[i][2][0]) + " " + str(trainYold[i][0]) +"\n")
			f_trainX,f_trainY,f_testX,f_testY = include_business_day('freeway_data/freeway_data2.csv')
			run_lstm(f_trainX,f_trainY,n_features,n_steps,epochs)
main()





