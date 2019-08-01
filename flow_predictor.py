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
from keras import optimizers
from keras import regularizers
import math
from sklearn import preprocessing



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
	    X.append(i[:3])
	    Y.append(i[3])
	return np.array(X), np.array(Y)

######################### VISUALIZATION #######################
def plot_loss(history, agg):
	fig = plt.figure()
	normie = preprocessing.normalize([history.history['loss']])
	print(normie)
	print(history.history['loss'])
	#plt.plot(normie[0])
	plt.plot(history.history['loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.show()
	fig.savefig("losses/"+str(agg)+"min.png",dpi=100)
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
	msk = np.random.rand(len(dt2)) < 0.7
	train_df = dt2[msk]
	test_df = dt2[~msk]

	dataset = dataset.drop(columns=["index"])
	
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
	return new_data

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



######################### SMOOTHING THE DATA ####################
#smoothing using std_dev and mean

def smooth_data(train_cp,filename):
    data = train_cp.iloc[:,0]
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


    
    for row in range(len(train_cp)):
        number_of_changes= 0
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
        #print("Number of changes: " + str(number_of_changes))
                
    new_d = data.values.reshape(len(data.values),1)
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
	x, y = create_dataset(dataset)
	return	x, y

def nn_model(trainX,trainY,params):
	model = Sequential()
	layer1 = Dense(64,input_dim=3, activation='relu',kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))
	layer2 = Dense(64, activation='relu',kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))
	layer3 = Dense(32, activation='relu',kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))
	#layer3 = Dense(400, activation='relu')
	model.add(layer1)
	model.add(layer2)
	#model.add(layer3)
	model.add(Dense(1))
	#optimizers 
	sgd = optimizers.SGD(lr=0.001)
	rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
	adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	
	#compiling the model
	model.compile(loss='mean_squared_error', optimizer=adam,metrics=['mse','mae','mape'])
	history= model.fit(trainX, trainY, epochs=200, verbose=0, batch_size=64,validation_split=0.2)
	
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
	if flag==0:
		obs_df = pd.read_csv(filename)
		obs = obs_df.values
		obs = obs.astype('float32')
		obsX, obsY = create_dataset(obs)
		pred = model.predict(obsX)
		mape = []
		for i in range(len(pred)):
			err = abs((pred[i]+1)-(obsY[i]+1))/abs(obsY[i]+1)
			if err > 1000:
				print(err)
				print("loool")
				break
			mape.append(err)
		m_mape= sum(mape)*100/(len(mape))
		#print("MAPE: ", m_mape)

		score = model.evaluate(obsX, obsY, verbose=0)
		#print("MAE:" ,score[2] ,"MSE: ", score[1])
		#plot_results(pred, obsY, "teste",1)
		return m_mape, score[2], score[1],score[3]
	else:
		pred = model.predict(filename)
		mape = []
		for i in range(len(pred)):
			err = abs((pred[i]+1)-(obsY[i]+1))/abs(obsY[i]+1)
			if err > 1000:
				print(err)
				print("loool")
				break
			mape.append(err)
		m_mape= sum(mape)*100/(len(mape))
		#print("MAPE: ", m_mape)

		score = model.evaluate(filename, obsY, verbose=0)
		#print("MAE:" ,score[2] ,"MSE: ", score[1])
		return m_mape, score[2], score[1],score[3]



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



def main():
	#print("primeira linha da main")
	#ID_Espira = input("Coloque id da espira: ")
	ID_Espira = "4_ct4"
	test_date = np.datetime64('2018-08-27')
	train_set, test_set = get_data(ID_Espira,0, test_date)
	
	#dt2, train_set, test_set, dataset = get_data(ID_Espira)
	train_cp = copy.deepcopy(train_set)
	test_cp = copy.deepcopy(test_set)
	
	#smoothing both sets
	smooth_train, df_train= smooth_data(train_cp, "train_2") 
	smooth_test, df_test = smooth_data(test_cp, "test_2")

	#interval = input("Escolha o horizonte de predicao (15,30,45 ou 60): ")
	#if not interval:
	#	interval = 15
	#for interval in [15,30,45,60]:

	#	cenas = aggregate_data(smooth_train,"train", interval)
	#	cenas = aggregate_data(smooth_test,"test",interval)
	#	transform_data("train_"+str(interval)+".csv","train_formatted.csv")
	#	transform_data("test_"+str(interval)+".csv","test_formatted.csv")

		#smoothing with z score
		#NOT IN USE
		#zscore_train = smooth_zscore(train_cp)

		#Original datasets before smoothing

	dataframe = pd.read_csv('train.csv')
	dataset = dataframe.drop(columns=["Data"])
	dataset = dataset.values
	train = dataset.astype('float32')

	dataframe = pd.read_csv('test.csv')
	dataset = dataframe.drop(columns=["Data"])
	dataset = dataset.values
	test = dataset.astype('float32')
	
	# plot differences between original and smoothed data
	plot_changes(train, df_train)
	#plot_changes(test, df_test)
	
	#prepares the data for the nn 
	#transform_data("test_2.csv", "3day_unsmoothed.csv",3) 

	#transform_data("test_2.csv", "test_formatted.csv")
	#transform_data("train_2.csv", "train_formatted.csv")
	
	transform_data("test_set.csv", "test_set2.csv") #specific day testing data 

	

	#transform_data("train_30min.csv","train_formatted.csv")
	#transform_data("test_30min.csv","test_formatted.csv")
	
	# creates and trains the model
	#trainX, trainY = get_nn_data('train_formatted.csv')
	#testX, testY = get_nn_data('test_formatted.csv')

	################################################################
	#freeway dataset
	# dataframe = pd.read_csv("freeway_data/freeway_data2.csv", usecols=[1], engine='python')
	# dataset = dataframe.values
	# dataset = dataset.astype('float32')

	# train_size = int(len(dataset) * 0.80)
	# test_size = len(dataset) - train_size
	# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	# print(len(train), len(test))


	for interval in [15,30,45,60]:

		train, test = freeway_preprocess("freeway_data/freeway_data2.csv",interval)
		#print("Freeway data length")
		print(len(train), len(test))

		trainX, trainY = freeway_dataset(train,3)
		testX, testY =freeway_dataset(test,3)

		###################################################################
		run = 0
		model = ""
		history = ""
		train_runs = [[],[],[]]
		test_runs = [[],[],[]]
		loltrain_runs = [[],[],[],[]]#alterar para 3
		loltest_runs = [[],[],[],[]]
		while run < 2:
			model, history= nn_model(trainX,trainY,"cenas") #Eventualmente dar a opcao de escolher os hiperparametros
			mape, mae, mse,rmape= evaluate_model(trainX, model, 1,trainY)
			loltrain_runs[0].append(mape)
			loltrain_runs[1].append(mae)
			loltrain_runs[2].append(mse)
			loltrain_runs[3].append(rmape)
			#mape, mae, mse = evaluate_model('train_formatted.csv',model)
			#train_runs[0].append(mape)
			#train_runs[1].append(mae)
			#train_runs[2].append(mse)
			
			mape, mae, mse,rmape = evaluate_model(testX, model, 1,testY)
			loltest_runs[0].append(mape)
			loltest_runs[1].append(mae)
			loltest_runs[2].append(mse)
			loltest_runs[3].append(rmape)

			#mape, mae, mse = evaluate_model('test_formatted.csv',model)
			#test_runs[0].append(mape)
			#test_runs[1].append(mae)
			#test_runs[2].append(mse)
			
			run +=1
		plot_loss(history,interval)
		print("----------------------------------------------")
		print("Data Aggregation interval: ",interval)
		print("Average of 3 runs:  ")
		
		# print("Train")
		# print("MAPE: ", sum(train_runs[0])/len(train_runs[0]))
		# print("MAE: ",sum(train_runs[1])/len(train_runs[1]))
		# print("MSE: ",sum(train_runs[2])/len(train_runs[2]))

		# print("Test")
		# print("MAPE: ", sum(test_runs[0])/len(test_runs[0]))
		# print("MAE: ",sum(test_runs[1])/len(test_runs[1]))
		# print("MSE: ",sum(test_runs[2])/len(test_runs[2]))


		print("Train")
		print("MAPE: ", sum(loltrain_runs[0])/len(loltrain_runs[0]))
		print("REAL MAPE: ", sum(loltrain_runs[3])/len(loltrain_runs[3]))
		print("MAE: ",sum(loltrain_runs[1])/len(loltrain_runs[1]))
		print("MSE: ",sum(loltrain_runs[2])/len(loltrain_runs[2]))

		print("Test")
		print("MAPE: ", sum(loltest_runs[0])/len(loltest_runs[0]))
		print("REAL MAPE: ", sum(loltest_runs[3])/len(loltest_runs[3]))
		print("MAE: ",sum(loltest_runs[1])/len(loltest_runs[1]))
		print("MSE: ",sum(loltest_runs[2])/len(loltest_runs[2]))



		print("----------------------------------------------")

		break

	#evaluate model
	#model_results(trainX, trainY, testX, testY,model)

	#plot the results for the first three days
	#FIXME HAVE TO CREATE SLIDING WINDOW FOR THIS NEW DATASET
	#evaluate_model('train_formatted.csv',model)
	#evaluate_model('test_formatted.csv',model)
		
	
	#plot_results (predicted, observed, output file name, flag to choose how many days to plot)
	#plot_results(pred, obsY, "cenas",3) #change filename dynamically
main()
