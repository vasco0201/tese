import os
import csv
from os import listdir
from os.path import isfile, join


def main():
    out_file=  csv.writer(open("dados_waze.csv","w", newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
    out_file.writerow(["Data", "Hora","Rua","Velocidade","Gravidade","lat1","long1","lat2","long2"])
    w = open("waze_output.txt", "r+")
    lines = w.readlines()
    for line in lines:
        data = line.split(",")
        if "de Redondo" in data[2]:
            date = data[0]
            time = data[1]
            street = data[2]
            speed = data[3]
            level= data[4]
            lat1= data[5][1:]
            long1= data[6][:-1]
            lat2= data[7][1:]
            long2 = data[8][:-2]
            print(lat1, long1)
            print("-----------------------------")
            print(lat2,long2)
            #out_file.writerow(["2018-01-01",data[1],t,data[2],lat1,long1,lat2,long2])
            
        
    
main()