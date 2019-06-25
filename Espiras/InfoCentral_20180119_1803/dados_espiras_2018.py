import os
import csv
from os import listdir
from os.path import isfile, join
import re

def get_files():
    curr_dir = os.getcwd()
    
    return [f for f in listdir(curr_dir) if isfile(join(curr_dir, f))]

def create_csv():
    out_file=  csv.writer(open("dados_camara.csv","w", newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
    list_hours = get_hours()
    header =["Data","Zona","Contadores","ID_Espira"] + list_hours
    out_file.writerow(header)
    return out_file

def get_hours():
    hours = []
    for i in range(0,24):
        hour = str(i)+ "h"
        for m in (0,15,30,45):
            if (m == 0):
                temp = hour + "00"
                hours.append(temp)
            else:
                temp = hour + str(m)
                hours.append(temp)
    return hours
    


def main():
    #days_log = open("days_log.txt", "w")
    broken_log = open("broken_log.txt", "w")
    empty_log = open("empty_log.txt", "w")
    out_file = create_csv()
    files = get_files()
    for filename in files:
        
        if "CT15Mn".lower() not in filename.lower():
            continue
        if (os.stat(filename).st_size == 0):
            empty_log.write("-------------------Ficheiro:" + filename + "-----------------"+"\n")
            continue
        #print(filename)
        f = open(filename,"r+")
        
        #if ("18.06" in filename):
        #print(filename)
        lines = f.readlines()
        
        for line in lines:
            data = line.split("|") #split the values by |
            try:
                data.remove("")
                
            except ValueError:
                broken_log.write("-------------------Ficheiro:" + filename + "-----------------"+ "\n")
                continue
            print(data[0])
            if not re.search("[0-9]*\-[0-9]*\-[0-9]*",data[0]): #if entry doesn't start with the date then it is broken
                broken_log.write("-------------------Ficheiro:" + filename + "-----------------"+ "\n")
                continue
            data= data[:-1] 
            #print(data)
            out_file.writerow(data)
                
main()
