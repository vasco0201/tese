import os
from os import listdir
from os.path import isfile, join
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt



f = open("CT15Mn-240317.07", "r")

flow_data = []
lines = f.readlines()
for l in lines:
	data = l.split("|")
	data.remove("")
	data = data[:-1]
	if data[3] == "CT4" or data[3] == "CT6":
		print(len(data))
		print(data)
		temp = [data[0],data[3]]
		for i in data[4:]:
			#print i
			i = int(i)
			temp.insert(len(temp),i)
		flow_data.insert(len(flow_data), temp)

print(flow_data[0][1])


def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

dts = [dt.strftime('%H:%M') for dt in 
       datetime_range(datetime(2016, 9, 1, 0), datetime(2016, 9, 1, 23, 59), 
       timedelta(minutes=15))]
#print len(dts)

fig = plt.figure(1)
plt.xticks(rotation=90)
ax = fig.add_subplot(111)
ax.plot(dts,flow_data[0][2:], label=flow_data[0][1])
ax.plot(dts,flow_data[1][2:], label=flow_data[1][1])
ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,prop={'size': 15},title="ID Espira")
ax.grid(True)
ax.set_title("Cruzamento 313 "+ flow_data[0][0])
fig.set_size_inches(18.5, 12.5)
fig.savefig('24-03-17_zona7.png', dpi=100)
