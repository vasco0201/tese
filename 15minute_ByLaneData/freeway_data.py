import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import math
def get_files():
    curr_dir = os.getcwd()
    
    return [f for f in listdir(curr_dir) if isfile(join(curr_dir, f))]

def avg_per_ts(data_np,ts_target):
    avg=[]
    for k in range(len(data_np)):
        ts = data_np[k][0]
        if (ts.hour == ts_target.hour) & (ts.minute == ts_target.minute):
            if not math.isnan(data_np[k][1]):
                avg.append(data_np[k][1])
    return sum(avg)/len(avg)
def get_data(filename,iteration):
	dataset = pd.read_sas(filename) #md2004_15min_2a.sas7bdat") #md2004_15min_4b.sas7bdat
	data_np = dataset.values
	#print(data_np)
	miss_values = 0
	for i in range(len(data_np)):
	    if (math.isnan(data_np[i][1])) or (data_np[i][1]==0):
	        miss_values +=1
	        past_values = len(data_np[:i])
	        if (i == (len(data_np)-1)):
	            new_value = (data_np[i-1][1]+data_np[i-2][1])/2
	            data_np[i][1] = new_value
	            
	        elif (i == 0):
	            new_value = (data_np[i+1][1]+data_np[i+2][1])/2
	            if (math.isnan(new_value)):
	                ts_target = data_np[i][0]
	                new_value = avg_per_ts(data_np,ts_target)
	            data_np[i][1] = new_value
	        else:
	            new_value = (data_np[i+1][1]+data_np[i-1][1])/2
	            if (math.isnan(new_value)):
	                ts_target = data_np[i][0]
	                new_value = avg_per_ts(data_np,ts_target)
	                #print("new value: ",new_value)
	                #print("old prev: ",data_np[i-1][1])
	                #print("old next: ",data_np[i+1][1])
	            data_np[i][1] = new_value
	        #if past_values >= 4:
	        #    temp = data_np[i-4:i]
	        #    temp_2 = temp[:,1]
	        #    avg = np.mean(temp_2)
	        #    data_np[i][1] = avg
	            #print("Changing Nan to :",avg)
	        #else:
	        #    temp = data_np[i-past_values:i][1]
	        #    temp_2 = temp[:,1]
	        #    avg = np.mean(temp_2)
	        #    data_np[i][1] = avg
	            #print("(ELSE) Changing Nan to :",avg)
	print("---------------------------------------------------------")
	print("file: ",filename)
	print("iteration:", iteration)
	print("Missing values: ",miss_values)
	print("Percentage of missing values :", 100*(miss_values/len(data_np)), "%")
	print("---------------------------------------------------------")
	for i in range(len(data_np)):
	    if (math.isnan(data_np[i][1])):
	        print("Nan")
	df_data = pd.DataFrame(data_np)
	df_data.columns = ['Data', 'Count']

	df_data.to_csv("../freeway_data/freeway_data" + str(iteration) + ".csv", sep=',', index=False)

	#dataset.to_csv("freeway_data1.csv",sep = ",", index=False)
	#dataset['Data'] = pd.to_datetime(dataset['Data'], infer_datetime_format=True)

def main():
	files = get_files()
	print(files)
	for i in range(len(files)):
		filename, file_extension = os.path.splitext(files[i])
		if file_extension != ".sas7bdat":
			continue
		get_data(files[i], i)
main()