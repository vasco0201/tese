import pandas as pd
import numpy as np
import copy
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import statistics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import math
from talos.model.layers import hidden_layers
import talos as ta
from talos import Evaluate
from talos import Deploy
from talos.model.normalizers import lr_normalizer



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
	data_folder = os.path.join(curr_dir, "CT15Mn-150818_101018", "dados_camara.csv") #newdata
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
	
	dt2.to_csv("limited_data.csv", sep= ',', index=False)
	#train_set
	train_df.to_csv("train.csv", sep= ',', index=False)

	#test_set
	test_df.to_csv("test.csv", sep= ',', index=False)

	#full set
	dataset.to_csv("dados_nn.csv", sep=',', index=False)
	#train_df.reset_index(drop=True)
	return dt2, train_df, test_df, dataset


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

def nn_model(params):
	#dataframe = pd.read_csv('new_f.csv') #raw data
	#dataset = dataframe.values
	#dataset = dataset.astype('float32')
	#dataframe = pd.read_csv('test.csv') #data generated by reusing the same day 50 times


	dataframe = pd.read_csv('train_formatted.csv')
	dataset = dataframe.values
	train_set = dataset.astype('float32')

	dataframe = pd.read_csv('test_formatted.csv')
	dataset = dataframe.values
	test_set = dataset.astype('float32')

	print(len(train_set))
	print(len(test_set))
	trainX, trainY = create_dataset(train_set)
	testX, testY = create_dataset(test_set)
	
	

	#model, history = create_model(trainX, trainY, testX, testY, params)
	
	return trainX, trainY, testX, testY#,model

def create_model(trainX, trainY, testX, testY, params):
	model = Sequential()
	#params['n_neurons']
	model.add(Dense(params['first_neuron'],input_dim=3, activation=params['activation']))
	model.add(Dropout(params['dropout']))

	#hidden_layers(model, params, 1)

	model.add(Dense(1))
	model.compile(loss=params['losses'],
                  # here we add a regulizer normalization function from Talos
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
                  metrics=['mae'])
	#model.compile(loss='mean_squared_error', optimizer=rmsprop,metrics=['mape', 'mae', 'mse'])
	history = model.fit(trainX, trainY, epochs=params['epochs'],batch_size=params['batch_size']
		,verbose=0)
	nf = open("results_hyper.txt","a+")
	nf.write(str(history)+ "\n")
	nf.close()
	return history, model

def model_results(trainX, trainY, testX, testY,model):
	
	# Estimate model performance
	trainScore = model.evaluate(trainX, trainY, verbose=0)
	#print(model.metrics_names)
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
def evaluate_instance(filename, model):
	obs_df = pd.read_csv(filename)
	obs = obs_df.values
	obs = obs.astype('float32')
	obsX, obsY = create_dataset(obs)
	pred = model.predict(obsX)
	print("----------------PRED-----------------------")
	print(pred)
	print("----------------OBSX-----------------------")
	print(obsX)
	score = model.evaluate(obsX, obsY, verbose=0)
	print("MAE:" ,score[2] ,"RMSE: ", score[3])
	#plot_results(pred, obsY, "teste",1)



def main():
	#print("primeira linha da main")
	#ID_Espira = input("Coloque id da espira: ")
	ID_Espira = "4_ct4"
	#test_date = np.datetime64('2018-08-27')
	#dt2, train_set, test_set, dataset = get_data(ID_Espira, 1, test_date)
	dt2, train_set, test_set, dataset = get_data(ID_Espira)
	train_cp = copy.deepcopy(train_set)
	test_cp = copy.deepcopy(test_set)
	
	#smoothing both sets
	smooth_train, df_train= smooth_data(train_cp, "train_2") 
	smooth_test, df_test = smooth_data(test_cp, "test_2")

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
	transform_data("dados_nn.csv", "new_f.csv")
	transform_data("dados_nn.csv", "3day_nn.csv", 3)
	#transform_data("train2.csv", "smoothed_data.csv") not needed anymore
	transform_data("test_2.csv", "test_formatted.csv")
	transform_data("test_2.csv", "3day_unsmoothed.csv",3)
	transform_data("train_2.csv", "train_formatted.csv")
	transform_data("test_set.csv", "test_set2.csv")
	# creates and trains the model
	#'shape':['brick','long_funnel'], 'optimizer': [Adam, Nadam, RMSprop],
	
	print("Antes do nn model")

	p = {'lr': [0.001,0.01],
     'first_neuron':[32, 64,128],
     'hidden_layers':[1, 2, 3, 4],
     'batch_size': [10,96],
     'epochs': [200,300,400,500],
     'dropout': [0, 0.05],
     'optimizer': [optimizers.RMSprop],
     'shapes':['brick'], #,'long_funnel'],
     'activation':['relu'],
     'losses': ['mean_squared_error']#, 'binary_crossentropy']
    }
	trainX, trainY, testX, testY= nn_model(p) #Eventualmente dar a opcao de escolher os hiperparametros

	t = ta.Scan(x=trainX,
            y=trainY,
            model=create_model,
            grid_downsample=0.20, 
            params=p,
            dataset_name='traffic_flow',
            experiment_no='3')


	#evaluate model
	#model_results(trainX, trainY, testX, testY,model)

	#plot the results for the first three days
	#obs_df = pd.read_csv('3day_unsmoothed.csv')
	#FIXME HAVE TO CREATE SLIDING WINDOW FOR THIS NEW DATASET
	#evaluate_instance('test_set2.csv',model)


	
	#plot_results (predicted, observed, output file name, flag to choose how many days to plot)
	#plot_results(pred, obsY, "cenas",3) #change filename dynamically
main()
