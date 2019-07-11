import os
from os import listdir
from os.path import isfile, join
def get_files():
	curr_dir = os.getcwd()
	#return [f for f in listdir(curr_dir) if isfile(join(curr_dir, f))]
	for path, subdirs, files in os.walk(curr_dir):
	    for name in files:
	    	print(os.path.join(path, name))
#lista = get_files()
#print(lista)
get_files()