import pandas as pd
import numpy as np
import copy
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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

curr_dir = os.getcwd()

def create_dataset(data):
    X,Y = [],[]
    for i in data:
        X.append(i[:3])
        Y.append(i[3])
    return np.array(X), np.array(Y)

def get_data(ID_Espira):
	ID_Espira = "4_ct4"
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

	# msk = np.random.rand(len(dataset_uid)) < 0.8
	# train_df = dataset_uid[msk]
	# test_df = dataset_uid[~msk]
	dataset_2 = dataset_uid.values
	train_size = int(len(dataset_2) * 0.80)
	test_size = len(dataset_2) - train_size
	train, test = dataset_2[0:train_size,:], dataset_2[train_size:len(dataset_2),:]
	
	#np.delete(train_df, np.s_[0], axis=1)
	#np.delete(test_df,np.s_[0], axis=1)
	
	train_df = pd.DataFrame(train)
	test_df = pd.DataFrame(test)

	train_df = train_df.drop(columns=[0])
	test_df = test_df.drop(columns=[0])
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


def smooth_data(train_cp,filename):
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
	model.add(LSTM(64, activation='relu', input_shape=(n_steps, n_features), kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01)))
	#model.add(LSTM(32, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01)))
	#return_sequences=True

	model.add(Dense(1))
	model.compile(optimizer=adam, loss='mse',metrics=['mse','mae','mape'])
	# fit model
	patience= int(round(n_epochs/3))
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

	history = model.fit(trainX, trainY, epochs=n_epochs, verbose=2,validation_split=0.20)
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
	if flag==0:
		obs_df = pd.read_csv(filename)
		obs = obs_df.values
		obs = obs.astype('float32')
		obsX, obsY = create_dataset(obs)
		obsX = obsX.reshape((obsX.shape[0], obsX.shape[1], n_features))
		pred = model.predict(obsX)
		mape = calc_mape(pred,obsY)
		print("MAPE: ",mape)
		score = model.evaluate(obsX, obsY, verbose=0)
		print("MAE:" ,score[2] ,"MSE: ", score[1])
		return score[2],score[1],score[3],mape
		
		#plot_results(pred, obsY, "teste",1)
	else:
		filename = filename.reshape((filename.shape[0], filename.shape[1], n_features))
		pred = model.predict(filename)
		mape = calc_mape(pred,obsY)
		print("MAPE: ",mape)
		score = model.evaluate(filename, obsY, verbose=0)
		print("MAE:" ,score[2] ,"MSE: ", score[1])
		return score[2],score[1],score[3],mape


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
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


	


def main():
	n_features = 1
	n_steps = 3
	epochs = 20
	
	x = input("1 - Espiras; 2 - Autoestradas :")

	if x and int(x)==1:
		ID_Espira = "4_ct4"
		train_set, test_set = get_data(ID_Espira)

		train_cp = copy.deepcopy(train_set)
		test_cp = copy.deepcopy(test_set)
		
		smooth_train, df_train= smooth_data(train_cp, "train_2")
		smooth_test, df_test = smooth_data(test_cp, "test_2")

		transform_data("lstm/train_2.csv","lstm/train_formatted.csv")	
		transform_data("lstm/test_2.csv","lstm/test_formatted.csv")

		f_trainX, f_trainY = get_nn_data('lstm/train_formatted.csv')
		f_testX, f_testY = get_nn_data('lstm/test_formatted.csv')


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
			if not (os.path.exists("lstm_loopsensor_results.csv")):
				out_file =  csv.writer(open("lstm_loopsensor_results.csv","a+", newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
				header =["Espira","nr_past_ts","epochs","hidden_layers","n_units","mae","rmse","mape","test_mae", "test_rmse", "test_mape"]
				out_file.writerow(header)
			else:
				out_file =  csv.writer(open("lstm_loopsensor_results.csv","a+", newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
			out_file.writerow([ID_Espira, n_steps, epochs, n_layers, n_units, mae, rmse, mape_approx, tmae, trmse, tmape_approx])

	else: ################## freeway dataset ###########################
		
		dataframe = pd.read_csv('freeway_data/freeway_data2.csv', usecols=[1], engine='python')
		dataset = dataframe.values
		dataset = dataset.astype('float32')

		# split into train and test sets

		train_size = int(len(dataset) * 0.80)
		test_size = len(dataset) - train_size
		train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
		print("Freeway data length")
		print(len(train), len(test))


		f_trainX, f_trainY = freeway_dataset(train,6)
		f_testX, f_testY =freeway_dataset(test,6)
		
		################# LSTM Network ##################################
		
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
			out_file.writerow(["freeway_data2.csv", n_steps, epochs, n_layers, n_units, mae, rmse, mape, tmae, trmse, tmape])





main()





