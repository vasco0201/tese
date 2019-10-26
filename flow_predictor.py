import pandas as pd
import numpy as np
import copy
import sys
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import statistics
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import regularizers
import math
from sklearn import preprocessing
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import gc
import pickle
from sklearn.metrics import mean_squared_error

from math import sqrt



curr_dir = os.getcwd()

########################## AUX FUNCTIONS ########################
def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

def create_dataset(data):
    X,Y = [],[]
    for i in data:
        X.append(i[:-1])
        Y.append(i[-1])
    return np.array(X), np.array(Y)

######################### VISUALIZATION #######################
def plot_loss(history, agg):

	fig = plt.figure()
	new_loss = []
	x_axis=[]
	for i in range(len(history.history['loss'])):
		if i%10==0:
			new_loss.append(history.history['loss'][i])
			x_axis.append(i)
	print(new_loss)

	normie = preprocessing.normalize([history.history['loss']])
	print(history.history['loss'])
	plt.plot(x_axis[1:],new_loss[1:])
	#plt.plot(history.history['loss'][1:])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper right')
	plt.show()
	fig.savefig("floss"+str(agg)+"min.png",dpi=100)
	plt.close(fig)

def plot_changes(original, smooth, filename="avg2stdev"):
	hours = [dt.strftime('%H:%M') for dt in datetime_range(datetime(2018, 8, 15, 0, 0), datetime(2018, 8, 15, 23, 59),timedelta(minutes=15))]
	fig = plt.figure(2)

	plt.xticks(rotation=90)
	ax = fig.add_subplot(311)
	ax.plot(hours,smooth[0], label="smoothed")
	ax.plot(hours,original[0], label="original")
	ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,prop={'size': 15})#,title="ID Espira")
	ax.grid(True)
	ax.set_title("Predicted vs Expected")
	ax.tick_params(axis='x', rotation=90)

	ax = fig.add_subplot(312)
	ax.plot(hours,smooth[1], label="smoothed")
	ax.plot(hours,original[1], label="obs")
	ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,prop={'size': 15})#,title="ID Espira")
	ax.grid(True)
	ax.set_title("Predicted vs Expected")
	ax.tick_params(axis='x', rotation=90)

	ax = fig.add_subplot(313)
	ax.plot(hours,smooth[2], label="smoothed")
	ax.plot(hours,original[2], label="original")
	ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,prop={'size': 15})#,title="ID Espira")
	ax.grid(True)
	ax.set_title("Predicted vs Expected")
	ax.tick_params(axis='x', rotation=90)
	fig.set_size_inches(40, 20)
	fig.savefig(curr_dir + "\\smoothing\\" + str(filename) + ".png", dpi=200)
	plt.close(fig)
	#plt.show()

def plot_results(pred, obsY, plot_name, flag):
	dts = [dt.strftime('%H:%M') for dt in datetime_range(datetime(2018, 8, 15, 0, 45), datetime(2018, 8, 15, 23, 59), timedelta(minutes=15))]
	fig = plt.figure(1)
	plt.xticks(rotation=90)
	if flag == 3:
		ax = fig.add_subplot(311)
		ax.plot(dts,pred[0:93], label="pred")
		ax.plot(dts,obsY[0:93], label="obs")
		ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,prop={'size': 15})#,title="ID Espira")
		ax.grid(True)
		ax.set_title("Predicted vs Expected")
		ax.tick_params(axis='x', rotation=90)

		ax2 = fig.add_subplot(312)
		ax2.plot(dts, pred[93:186], label="pred")
		ax2.plot(dts,obsY[93:186], label="obs")
		ax2.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,prop={'size': 15})#,title="ID Espira")
		ax2.grid(True)
		ax2.set_title("Predicted vs Expected")
		ax2.tick_params(axis='x', rotation=90)

		ax3 = fig.add_subplot(313)
		ax3.plot(dts, pred[186:279], label="pred")
		ax3.plot(dts,obsY[186:279], label="obs")
		ax3.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,prop={'size': 15})#,title="ID Espira")
		ax3.grid(True)
		ax3.set_title("Predicted vs Expected")
		ax3.tick_params(axis='x', rotation=90)
	elif flag == 1:
		ax = fig.add_subplot(111)
		ax.plot(dts,pred[0:93], label="pred")
		ax.plot(dts,obsY[0:93], label="obs")
		ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,prop={'size': 15})#,title="ID Espira")
		ax.grid(True)
		ax.set_title("Predicted vs Expected")
		ax.tick_params(axis='x', rotation=90)

	fig.set_size_inches(40, 20)
	fig.savefig(curr_dir + "\\results_nn\\" +str(plot_name)+".png", dpi=100)
#

######################### READING THE DATA ####################

def get_data(ID_Espira, flag_test =0, test_date = np.datetime64('2018-08-27')):
	#data_folder = os.path.join(curr_dir, "CT15Mn-150818_101018", "dados_camara.csv")
	data_folder = os.path.join(curr_dir, "dados_camara_todos.csv")

	dataset = pd.read_csv(data_folder)#dados mais recentes
	#tomtom = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\tomtom_data.csv")
	#dataset = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\periodic_data.csv") #dados periodicos gerados automaticamente
	#dataset = pd.read_csv("\\Users\\ASUS\\Documents\\IST\\5ºAno\\dados_old.csv") #dados mais antigos
	dataset['unique_id'] = dataset.Zona.astype(str) + '_' + dataset.ID_Espira.astype(str)
	dataset['unique_id'] = dataset['unique_id'].str.lower()
	dataset = dataset[dataset["unique_id"] == str(ID_Espira)]

	dataset = dataset.drop(columns=["Zona","Contadores","ID_Espira","unique_id"])
	dt2 = copy.deepcopy(dataset)
	#dataset.sort_values(['Data'],ascending=True).groupby('Data').reset_index()
	dataset = dataset.groupby('Data').apply(lambda x: x.reset_index())
	msk = np.random.rand(len(dt2)) < 0.8
	train_df = dt2[msk]
	test_df = dt2[~msk]

	dataset = dataset.drop(columns=["index"])
	#print(dataset.head())
	if flag_test == 1:
		#creating a new testing data set for comparison with Sarima Model or other models
		test_set = dataset[(dataset['Data'] == str(test_date))]
		test_set.to_csv("test_set.csv", sep= ',', index=False)

		train_df = copy.deepcopy(dataset)
		test_df = copy.deepcopy(dataset)

		for index, row in dataset.iterrows():
			date = np.datetime64(row['Data'])
			if date < test_date:
				test_df[test_df.Data != row['Data']]
			else :
				train_df[train_df.Data != row['Data']]
		#data = pd.to_datetime(dataset['Data'], infer_datetime_format=True)
		#train_df = dataset[()]
		#test_df  = dataset[(data >= test_date)]
		#sys.exit()
	#train_set

	train_df.to_csv("train.csv", sep= ',', index=False)

	#test_set
	test_df.to_csv("test.csv", sep= ',', index=False)

	return train_df, test_df
def aggregate_data(dataset,filename, interval):
	new_data = []
	for i in range(len(dataset)):
		j = 1
		temp=[]
		if (int(interval) == 15):
			
			pd.DataFrame(dataset).to_csv(str(filename) + "_15.csv", sep=',', index=False)
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
				pd.DataFrame(dataset).to_csv(str(filename) + "_15.csv", sep=',', index=False)
				return dataset
		temp.insert(0,dataset[i][0])
		new_data.append(temp)
	new_data = np.asarray(new_data)
	pd.DataFrame(new_data).to_csv(str(filename) + "_" + str(interval) + ".csv", sep=',', index=False)

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




######################### SMOOTHING THE DATA ####################
#smoothing using std_dev and mean
def smooth_data(train_cp,filename):
    data = train_cp.iloc[:,0]
    data_np = data.values
    #print(data.values)
    train_cp = train_cp.drop(columns=["Data"])
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
    print("Numero de dias inuteis:", len(to_delete))
    new_d = data_np.reshape(len(data_np),1)
    test = np.concatenate((new_d,train_cp),axis=1)
    pd.DataFrame(test).to_csv(str(filename) + ".csv", sep=',', index=False)
    return test, train_cp


def smooth_zscore(train2):
	#smoothing using z_score
	#not great
	
	#test_cp = copy.deepcopy(test_df)
	date = copy.deepcopy(train2.iloc[:,0])
	values = copy.deepcopy(train2.iloc[:,1:])
	cenas = values.values

	z = np.abs(stats.zscore(values))
	y = np.abs(stats.zscore(cenas))
	threshold = 2
	z_score = np.where(y > 3)
	#print(cenas[1])
	#print(cenas[1,-2])
	#print(train2.columns[9])
	#z_score doesnt really work very well
	for i in z_score[0]:
	    for j in z_score[1]:
	        #dois casos especiais : 0h00 e 23h45
	        if j == 0:
	            #values.columns[j] == "0h00":
	            #dostuff
	            previous_t = cenas[i,-1]
	            
	            next_t = cenas[i,j+1]
	            cenas[i,j] = (previous_t + next_t)/2
	        if j == 95:
	            #dostuff
	            next_t = cenas[i,0]
	            previous_t = cenas[i,j-1]
	            cenas[i,j] = (previous_t + next_t)/2
	        else:
	            previous_t = cenas[i,j-1]
	            next_t = cenas[i,j+1]
	            cenas[i,j] = (previous_t + next_t)/2


	return cenas            #values.loc[i,values.columns[j]] = (previous_t + next_t)/2
def get_nn_data(filename):
	dataframe = pd.read_csv(str(filename))
	dataset = dataframe.values
	dataset = dataset.astype('float32')
	#print(dataset[:5])
	x, y = create_dataset(dataset)
	#print("X:",x[:5])
	#print("Y:",y[:5])
	return	x, y

def nn_model(trainX,trainY,params,dim_input,n_epochs=200):
	model = Sequential()
	layer1 = Dense(64,input_dim=dim_input, activation='relu',kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))
	layer2 = Dense(64, activation='relu',kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))
	layer3 = Dense(64, activation='relu',kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))
	layer4 = Dense(64, activation='relu',kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))
	layer5 = Dense(64, activation='relu',kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))
	
	#layer3 = Dense(400, activation='relu')
	model.add(layer1)
	model.add(layer2)
	#model.add(layer3)
	#model.add(layer4)
	#model.add(layer5)

	model.add(Dense(1))
	#optimizers 
	sgd = optimizers.SGD(lr=0.001)
	rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
	adam=optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	
	#compiling the model
	model.compile(loss='mean_squared_error', optimizer=adam,metrics=['mse','mae','mape'])
	patience= int(round(n_epochs/3))
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
	history= model.fit(trainX, trainY, epochs=n_epochs, verbose=0, batch_size=64,validation_split=0.2, callbacks=[es])
	
	return model,history


def model_results(trainX, trainY, testX, testY,model):
	
	# Estimate model performance
	trainScore = model.evaluate(trainX, trainY, verbose=0)
	print(model.metrics_names)
	#print(trainScore)
	#print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
	testScore = model.evaluate(testX, testY, verbose=0)
	#print(testScore)

	#print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

	# generate predictions for training
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	#print(trainPredict)
	#print(trainY)
	#print(testPredict)
	#print(testY)
	layers_conf = []
	units = []
	activation = []
	for i in range(len(model.layers)):
	    layer = model.layers[i].get_config()
	    layers_conf.append(layer)
	    units.append(layer["units"])
	    activation.append(layer["activation"])
	if (os.path.isfile('metrics_comparison.txt')) and (os.stat("metrics_comparison.txt").st_size == 0):
	    f = open("metrics_comparison.txt", "w+")
	    f.write("N_layers,[units per layer], [activations]," + "\n")
	    f.write("Metrics names: "  + str(model.metrics_names)+ "\n\n")
	    
	    f.write(str(len(layers_conf)) + "," + str(units) + "," +  str(activation) +"\n")
	    f.write("Train Set metrics :"  + str(trainScore) +"\n")
	    f.write("Test Set metrics :"  + str(testScore) +"\n\n\n")
	    f.close()
	else:
	    f = open("metrics_comparison.txt", "a+")
	    f.write(str(len(layers_conf)) + "," + str(units) + "," +  str(activation) +"\n")
	    f.write("Train Set metrics :"  + str(trainScore) +"\n")
	    f.write("Test Set metrics :"  + str(testScore) +"\n\n\n")
	    f.close()
def evaluate_model(filename, model,flag=0, obsY="data"):
	to_delete=[]
	for i in range(len(obsY)):
		if obsY[i] == 0:
			to_delete.append(i)

	obs_withoutZero = np.delete(obsY,to_delete, axis=0)
	filename_without_zero = np.delete(filename,to_delete,axis=0)

	obs_onlyZero = obsY[to_delete]
	filename_only_zero = filename[to_delete]
	
	score = model.evaluate(filename, obsY, verbose=0)
	rmse_total = sqrt(score[1])
	if len(filename_only_zero) != 0:
		score_no_zeros = model.evaluate(filename_without_zero, obs_withoutZero, verbose=0)
		score_of_zeros = model.evaluate(filename_only_zero, obs_onlyZero, verbose=0)
		return score_no_zeros[3], score_no_zeros[2], sqrt(score_no_zeros[1]), score_of_zeros[2], sqrt(score_of_zeros[1]), score[0], rmse_total, score[2]
	else:
		return score[3], 1, 1, 1, 1, score[0], rmse_total, score[2]
	#print("Calculated through model evaluate")
	#print("RMSE:",sqrt(scoreWithoutzeros[1]), "MAE:", scoreWithoutzeros[2], "MAPE:", scoreWithoutzeros[3])

	#print("MAE:" ,score[2] ,"MSE: ", score[1])


################## FREEWAY DATA ########################
def freeway_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
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
#function that plots the variation of mape with the
#increase in the number of past timeteps considered in the network
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
	fig.savefig("/home/vasco/Desktop/urban_plots/"+str(title),dpi=100)
	#plt.subplots_adjust(top=0.935,bottom=0.145,left=0.065,right=0.989,hspace=0.2,wspace=0.2)
	#plt.tight_layout()
	#plt.subplots_adjust(bottom=0.19)
	plt.show()

	plt.close()
############################# TESTS ################################




	
def mape_vs_timealag(filename,epochs):
	with open("trainMAPE_prev_ts.txt", "rb") as fp:   # Unpickling
		train_mape_evo = pickle.load(fp)
	with open("testMAPE_prev_ts.txt", "rb") as fp:   # Unpickling
		test_mape_evo = pickle.load(fp)
	i = 4
	for interval in [5,6,7]:
		print("------------------------------------------------------------------")
		print("Tou no:", interval)
		print("------------------------------------------------------------------")
		train, test = freeway_preprocess(filename,15)
		print("Freeway data length")
		print(len(train), len(test))

		trainX, trainY = freeway_dataset(train,interval)
		testX, testY =freeway_dataset(test,interval)
		print(len(trainX[0]))
		model, history= nn_model(trainX,trainY,"cenas",interval,epochs)
		
		mape, mae, mse,rmape,loss = evaluate_model(trainX, model, 1,trainY)
		train_mape_evo[i].append(mape)
		tmape, tmae, tmse,trmape,tloss = evaluate_model(testX, model, 1,testY)
		test_mape_evo[i].append(tmape)
		gc.collect()
		i+=1
	#print(train_mape_evo)
	#print(test_mape_evo)
	with open("trainMAPE_prev_ts.txt", "wb") as fp:
 		pickle.dump(train_mape_evo, fp)
	with open("testMAPE_prev_ts.txt", "wb") as fp:
		pickle.dump(test_mape_evo, fp)
	print(i)
def loss_nlayers(filename,epochs):
	with open("trainloss_nlayers.txt", "rb") as fp:   # Unpickling
		loss_lst = pickle.load(fp)
	with open("testloss_nlayers.txt", "rb") as fp:   # Unpickling
		testloss_lst = pickle.load(fp)
	interval = 4
	train, test = freeway_preprocess(filename,15)
	trainX, trainY = freeway_dataset(train,interval)
	testX, testY =freeway_dataset(test,interval)
	model, history= nn_model(trainX,trainY,"cenas",interval,epochs)
	mape, mae, mse,rmape,loss = evaluate_model(trainX, model, 1,trainY)
	tmape, tmae, tmse,trmape,tloss = evaluate_model(testX, model, 1,testY)
	loss_lst.append(loss)
	testloss_lst.append(tloss)
	with open("trainloss_nlayers.txt", "wb") as fp:
 		pickle.dump(loss_lst, fp)
	with open("testloss_nlayers.txt", "wb") as fp:
		pickle.dump(testloss_lst, fp)
	plot_error(loss_lst,testloss_lst,"loss_nlayers")


def run_multi_aggregation(filename,epochs,n_steps):
	#f = open("lstm_mape_multi_aggregation.csv","a+")
	#f.write("15,30,45,60\n")
	
	#g = open("test_lstm_mape_multi_aggregation.csv","a+")
	#g.write("15,30,45,60\n")
	
	train_mape_file = open("/content/drive/My Drive/freeway/freeway_train_mape_agg.txt", "wb")


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

	#h = open("lstm_mape_multi_aggregation.csv","a+")
	#h.write("15,30,45,60\n")
	for run in range(5):
		i=0
		print("Run nr:", run+1,"/5")
		for interval in [15,30,45,60]:
			train,test = freeway_preprocess(filename,interval)

			trainX, trainY = freeway_dataset(train,n_steps)
			testX, testY =freeway_dataset(test,n_steps)

			model, history= nn_model(trainX,trainY,"cenas",n_steps,epochs)
			mape, mae, rmse, mae_zeros, rmse_zeros, loss, rmse_total, mae_total = evaluate_model(trainX, model, 1,trainY)
			tmape, tmae, trmse, tmae_zeros, trmse_zeros, tloss, trmse_total, tmae_total = evaluate_model(testX, model, 1,testY)

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

			i+=1

		print("Saving progress...")

		train_mape_file = open("/content/drive/My Drive/freeway/freeway_train_mape_agg.txt", "wb")
		test_mape_file = open("/content/drive/My Drive/freeway/freeway_test_mape_agg.txt", "wb")

		train_rmse_file = open("/content/drive/My Drive/freeway/freeway_train_rmse_agg.txt", "wb")
		test_rmse_file = open("/content/drive/My Drive/freeway/freeway_test_rmse_agg.txt", "wb")

		train_mae_file = open("/content/drive/My Drive/freeway/freeway_train_mae_agg.txt", "wb")
		test_mae_file =	open("/content/drive/My Drive/freeway/freeway_test_mae_agg.txt", "wb")

		train_rmse_zeros_file  = open("/content/drive/My Drive/freeway/freeway_train_rmseWzeros_agg.txt", "wb")
		test_rmse_zeros_file = open("/content/drive/My Drive/freeway/freeway_test_rmseWzeros_agg.txt", "wb")
		
		train_mae_zeros_file = open("/content/drive/My Drive/freeway/freeway_train_maeWzeros_agg.txt", "wb")
		test_mae_zeros_file = open("/content/drive/My Drive/freeway/freeway_test_maeWzeros_agg.txt", "wb")
		
		train_rmse_total_file = open("/content/drive/My Drive/freeway/freeway_train_rmsetotal_agg.txt", "wb")
		test_rmse_total_file = open("/content/drive/My Drive/freeway/freeway_test_rmsetotal_agg.txt", "wb")

		train_mae_total_file = open("/content/drive/My Drive/freeway/freeway_train_maetotal_agg.txt", "wb")
		test_mae_total_file = open("/content/drive/My Drive/freeway/freeway_test_maetotal_agg.txt", "wb")
		
		pickle.dump(train_mape_agg,train_mape_file)
		pickle.dump(test_mape_agg,test_mape_file)

		pickle.dump(train_rmse_agg,train_rmse_file)
		pickle.dump(test_rmse_agg,test_rmse_file)

		pickle.dump(train_mae_agg,train_mae_file)
		pickle.dump(test_mae_agg,test_mae_file)

		pickle.dump(train_rmse_zeros_agg,train_rmse_zeros_file)
		pickle.dump(test_rmse_zeros_agg,test_rmse_zeros_file)

		pickle.dump(train_mae_zeros_agg,train_mae_zeros_file)
		pickle.dump(test_mae_zeros_agg,test_mae_zeros_file)

		pickle.dump(train_rmse_total_agg,train_rmse_total_file)
		pickle.dump(test_rmse_total_agg,test_rmse_total_file)

		pickle.dump(train_mae_total_agg,train_mae_total_file)
		pickle.dump(test_mae_total_agg,test_mae_total_file)



		train_mape_file.close()
		test_mape_file.close()
		train_rmse_file.close()
		test_rmse_file.close()
		train_mae_file.close()
		test_mae_file.close()
		train_rmse_zeros_file.close()
		test_rmse_zeros_file.close()
		train_mae_zeros_file.close()
		test_mae_zeros_file.close()
		train_rmse_total_file.close()
		test_rmse_total_file.close()
		train_mae_total_file.close()
		test_mae_total_file.close()

def urban_multi_agg(train,test,n_steps=5):
	
	train_mape_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_train_mape_agg.txt", "wb")
	train_mape_file.close()

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


	for run in range(4):
		print("Run nr:", run+1,"/3")
		i=0
		for interval in [15,30,45,60]:
			print("Agg:", interval)
			aggregate_data(train,"train", interval)
			aggregate_data(test,"test",interval)
			transform_data2("train_"+str(interval)+".csv","train_formatted.csv",n_steps)
			transform_data2("test_"+str(interval)+".csv","test_formatted.csv",n_steps)

			trainX, trainY = get_nn_data('train_formatted.csv')
			testX, testY = get_nn_data('test_formatted.csv')
			
			model, history= nn_model(trainX,trainY,"cenas",n_steps-1,450) #Eventualmente dar a opcao de escolher os hiperparametros
		
			mape, mae, rmse, mae_zeros, rmse_zeros, loss, rmse_total, mae_total = evaluate_model(trainX, model, 1,trainY)
			tmape, tmae, trmse, tmae_zeros, trmse_zeros, tloss, trmse_total, tmae_total = evaluate_model(testX, model, 1,testY)
			
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

		train_mape_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_train_mape_agg.txt", "wb")
		test_mape_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_test_mape_agg.txt", "wb")

		train_rmse_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_train_rmse_agg.txt", "wb")
		test_rmse_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_test_rmse_agg.txt", "wb")

		train_mae_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_train_mae_agg.txt", "wb")
		test_mae_file =	open("/content/drive/My Drive/4_ct6/agg_test/urban_test_mae_agg.txt", "wb")

		train_rmse_zeros_file  = open("/content/drive/My Drive/4_ct6/agg_test/urban_train_rmseWzeros_agg.txt", "wb")
		test_rmse_zeros_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_test_rmseWzeros_agg.txt", "wb")
		
		train_mae_zeros_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_train_maeWzeros_agg.txt", "wb")
		test_mae_zeros_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_test_maeWzeros_agg.txt", "wb")
		
		train_rmse_total_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_train_rmsetotal_agg.txt", "wb")
		test_rmse_total_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_test_rmsetotal_agg.txt", "wb")

		train_mae_total_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_train_maetotal_agg.txt", "wb")
		test_mae_total_file = open("/content/drive/My Drive/4_ct6/agg_test/urban_test_maetotal_agg.txt", "wb")
		
		pickle.dump(train_mape_agg,train_mape_file)
		pickle.dump(test_mape_agg,test_mape_file)

		pickle.dump(train_rmse_agg,train_rmse_file)
		pickle.dump(test_rmse_agg,test_rmse_file)

		pickle.dump(train_mae_agg,train_mae_file)
		pickle.dump(test_mae_agg,test_mae_file)

		pickle.dump(train_rmse_zeros_agg,train_rmse_zeros_file)
		pickle.dump(test_rmse_zeros_agg,test_rmse_zeros_file)

		pickle.dump(train_mae_zeros_agg,train_mae_zeros_file)
		pickle.dump(test_mae_zeros_agg,test_mae_zeros_file)

		pickle.dump(train_rmse_total_agg,train_rmse_total_file)
		pickle.dump(test_rmse_total_agg,test_rmse_total_file)

		pickle.dump(train_mae_total_agg,train_mae_total_file)
		pickle.dump(test_mae_total_agg,test_mae_total_file)



		train_mape_file.close()
		test_mape_file.close()
		train_rmse_file.close()
		test_rmse_file.close()
		train_mae_file.close()
		test_mae_file.close()
		train_rmse_zeros_file.close()
		test_rmse_zeros_file.close()
		train_mae_zeros_file.close()
		test_mae_zeros_file.close()
		train_rmse_total_file.close()
		test_rmse_total_file.close()
		train_mae_total_file.close()
		test_mae_total_file.close()

		print("Saved!")


def urban_mape_timelag(agg):
	train_mape_evo = [[],[],[],[],[],[],[],[],[],[]]
	test_mape_evo = [[],[],[],[],[],[],[],[],[],[]]

	train_rmse_evo = [[],[],[],[],[],[],[],[],[],[]]
	test_rmse_evo = [[],[],[],[],[],[],[],[],[],[]]


	train_mae_evo = [[],[],[],[],[],[],[],[],[],[]]
	test_mae_evo = [[],[],[],[],[],[],[],[],[],[]]

	train_mape_file = open("/content/drive/My Drive/4_ct6/ts_test/urban_train_mape_ts.txt", "wb")
	train_mape_file.close()
	for run in range(3):
		i = 0
		print("Run:", run+1,"/3")
		for lag in range(2,12):
			print("Lag:", lag-1, "/10")
			transform_data2("train_"+str(agg)+".csv","train_formatted.csv",lag)
			transform_data2("test_"+str(agg)+".csv","test_formatted.csv",lag)
			

			trainX, trainY = get_nn_data('train_formatted.csv')
			testX, testY = get_nn_data('test_formatted.csv')
			model, history= nn_model(trainX,trainY,"cenas",lag-1,450) #Eventualmente dar a opcao de escolher os hiperparametros
			mape, mae, rmse, mae_zeros, rmse_zeros, loss, rmse_total, mae_total  = evaluate_model(trainX, model, 1,trainY)
			tmape, tmae, trmse, tmae_zeros, trmse_zeros, tloss, trmse_total, tmae_total  = evaluate_model(testX, model, 1,testY)
			train_mape_evo[i].append(mape)
			test_mape_evo[i].append(tmape)

			train_rmse_evo[i].append(rmse_total)
			test_rmse_evo[i].append(trmse_total)

			train_mae_evo[i].append(mae_total)
			test_mae_evo[i].append(tmae_total)

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
		train_mape_file = open("/content/drive/My Drive/4_ct6/ts_test/urban_train_mape_ts.txt", "wb")
		test_mape_file = open("/content/drive/My Drive/4_ct6/ts_test/urban_test_mape_ts.txt", "wb")

		train_rmse_file = open("/content/drive/My Drive/4_ct6/ts_test/urban_train_rmse_ts.txt", "wb")
		test_rmse_file = open("/content/drive/My Drive/4_ct6/ts_test/urban_test_rmse_ts.txt", "wb")

		train_mae_file = open("/content/drive/My Drive/4_ct6/ts_test/urban_train_mae_ts.txt", "wb")
		test_mae_file = open("/content/drive/My Drive/4_ct6/ts_test/urban_test_mae_ts.txt", "wb")
		
		pickle.dump(train_mape_evo,train_mape_file)
		pickle.dump(test_mape_evo,test_mape_file)

		pickle.dump(train_rmse_evo,train_rmse_file)
		pickle.dump(test_rmse_evo,test_rmse_file)
		
		pickle.dump(train_mae_evo,train_mae_file)
		pickle.dump(test_mae_evo,test_mae_file)

		train_mape_file.close()
		test_mape_file.close()
		train_rmse_file.close()
		test_rmse_file.close()

		print("Saved!")









def main():
	n_steps=4
	epochs=600
	#ID_Espira = input("Coloque id da espira: ")
	#ID_Espira = "4_ct4"
	ID_Espira = "4_ct6"
	test_date = np.datetime64('2018-08-27')
	train_set, test_set = get_data(ID_Espira,0, test_date)
	#dt2, train_set, test_set, dataset = get_data(ID_Espira)
	train_cp = copy.deepcopy(train_set)
	test_cp = copy.deepcopy(test_set)

	
	#smoothing both sets_
	smooth_train, df_train= smooth_data(train_cp, "train_2")
	smooth_test, df_test = smooth_data(test_cp, "test_2")
	#interval = input("Escolha o horizonte de predicao (15,30,45 ou 60): ")
	#if not interval:
	interval = 15
	
	smooth_train = pd.read_csv("train_2.csv")
	smooth_train = smooth_train.values


	smooth_test = pd.read_csv("test_2.csv")
	smooth_test = smooth_test.values

	aggregate_data(smooth_train,"train", interval)
	aggregate_data(smooth_test,"test",interval)



	#transform_data("train_"+str(interval)+".csv","train_formatted.csv")
	#transform_data("test_"+str(interval)+".csv","test_formatted.csv")

	#mape_with varying time lag size
	#urban_mape_timelag(interval)

	# #mape with varying prediction
	urban_multi_agg(smooth_train,smooth_test,5)

	# transform_data2("train_"+str(interval)+".csv","train_formatted.csv",5)
	# transform_data2("test_"+str(interval)+".csv","test_formatted.csv",5)
	# trainX, trainY = get_nn_data('train_formatted.csv')
	# testX, testY = get_nn_data('test_formatted.csv')
	# model, history= nn_model(trainX,trainY,"cenas",4,450) #Eventualmente dar a opcao de escolher os hiperparametros
	
	# mape, mae, rmse, mae_zeros, rmse_zeros, loss, rmse_total, mae_total = evaluate_model(trainX, model, 1,trainY)
	# tmape, tmae, trmse, tmae_zeros, trmse_zeros, tloss, trmse_total, tmae_total = evaluate_model(testX, model, 1,testY)
	# print("TRAIN:")
	# print("obs != 0:\nMAPE:",mape,"MAE:",mae, "RMSE:",rmse)
	# print("obs = 0:\nMAPE:","NA","MAE:",mae_zeros, "RMSE:",rmse_zeros)
	# print("ALL:\nMAPE:  NA","MAE:",mae_total, "RMSE:",rmse_total)


	# print("\n\nTEST:")
	# print("obs != 0:\nMAPE:",tmape,"MAE:",tmae, "RMSE:",trmse)
	# print("obs = 0:\nMAPE:","NA","MAE:",tmae_zeros, "RMSE:",trmse_zeros)
	# print("ALL:\nMAPE:  NA","MAE:",tmae_total, "RMSE:",trmse_total)
	#print(mape_approx)
	#print(tmape_approx)
	
	#print(rmape)
	#print(trmape)
	# # plot_loss(history,15)



	# avg_train_mape_evo= []
	# avg_test_mape_evo= []

	# with open("Urban_train_mape_multi_agg.txt", "rb") as fp:   # Unpickling
	# 	train_mape_evo = pickle.load(fp)
	# with open("Urban_test_mape_multi_agg.txt", "rb") as fp:   # Unpickling
	# 	test_mape_evo = pickle.load(fp)

	# for i in range(len(train_mape_evo)):
	# 	avg_train_mape_evo.append((sum(train_mape_evo[i])/len(train_mape_evo[i]))[0])
	# 	avg_test_mape_evo.append((sum(test_mape_evo[i])/len(test_mape_evo[i]))[0])
	# print(avg_train_mape_evo)
	# print(avg_test_mape_evo)


	# plot_error(avg_train_mape_evo,avg_test_mape_evo,"pred_horizon")


	#smoothing with z score
	#NOT IN USE
	#zscore_train = smooth_zscore(train_cp)

	
	# plot differences between original and smoothed data
	#plot_changes(train, df_train)
	#plot_changes(test, df_test)
	
	#prepares the data for the nn 
	#transform_data("test_2.csv", "3day_unsmoothed.csv",3) 

	#transform_data("test_2.csv", "test_formatted.csv")
	#transform_data("train_2.csv", "train_formatted.csv")
	
	#transform_data("test_set.csv", "test_set2.csv") #specific day testing data 

	

	#transform_data("train_30min.csv","train_formatted.csv")
	#transform_data("test_30min.csv","test_formatted.csv")
	
	#creates and trains the model
	#trainX, trainY = get_nn_data('train_formatted.csv')
	#testX, testY = get_nn_data('test_formatted.csv')
		

	###############################################################
	#freeway dataset
	# print("Tou no sitio certo")
	# n_steps = 5
	# train,test = freeway_preprocess("4_ct4_timeseries2013.csv",15)
	# "freeway_data/freeway_data2.csv"
	# #minitest = test[:96]
	# #print(minitest)
	# trainX, trainY = freeway_dataset(train,n_steps)
	# testX, testY =freeway_dataset(test,n_steps)

	# model, history= nn_model(trainX,trainY,"cenas",n_steps,epochs)

	# mape, mae, rmse, mae_zeros, rmse_zeros, loss, rmse_total, mae_total = evaluate_model(trainX, model, 1,trainY)
	# tmape, tmae, trmse, tmae_zeros, trmse_zeros, tloss, trmse_total, tmae_total  = evaluate_model(testX, model, 1,testY)

	# #tmape_approx, tmae, tmse,tmape,tloss = evaluate_model(testX, model, 1,testY)

	# #print("Train Mape: ", mape)
	# #print("Test Mape: ", tmape)

	# print("Train:")
	# print("MAPE:", mape, "RMSE:", rmse_total, "MAE:", mae_total)

	# print("Test:")
	# print("MAPE:", tmape, "RMSE:", trmse_total, "MAE:", tmae_total)



	#after all the tests are done load the lists and compute avg
	# avg_train_mape_evo= []
	# avg_test_mape_evo= []

	# for i in range(len(train_mape_evo)):
	# 	print(len(train_mape_evo[i]))
	# 	avg_train_mape_evo.append(sum(train_mape_evo[i])/len(train_mape_evo[i]))
	# 	avg_test_mape_evo.append(sum(test_mape_evo[i])/len(test_mape_evo[i]))

	# print(avg_train_mape_evo)
	# print(avg_test_mape_evo)
	# train_mape_evo = [[],[],[],[],[],[]] 
	# test_mape_evo =	[[],[],[],[],[],[]]

	
	######## tests ############
	#evolution of mape with the increase in the input size
	#mape_vs_timealag("freeway_data/freeway_data2.csv",600)

	#variation of loss with n_hidden layers
	#loss_nlayers("freeway_data/freeway_data2.csv",600)

	#variation of pred horizon
	#run_multi_aggregation("freeway_data/freeway_data2.csv",600,6)

	






	###################################################################
	# print("----------------------------------------------")

	#evaluate model
	#model_results(trainX, trainY, testX, testY,model)

		
	
	#plot_results (predicted, observed, output file name, flag to choose how many days to plot)
	#plot_results(pred, obsY, "cenas",3) #change filename dynamically
main()
