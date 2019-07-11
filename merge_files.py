#small script to merge files
import sys
import os
import pandas as pd
files = []

if (len(sys.argv))>1:
	for i in sys.argv[1:]:
		files.append(pd.read_csv(i))
else:
	files = [pd.read_csv('/home/vasco/tese/CT15Mn-111018-150419/dados_camara.csv'),pd.read_csv('/home/vasco/tese/CT15Mn-150818_101018/dados_camara.csv'), pd.read_csv('/home/vasco/tese/Espiras/InfoCentral_20180119_1803/dados_camara.csv')]
combined_csv = pd.concat(files)
#export to csv
combined_csv.to_csv("dados_camara_todos.csv", sep= ',', index=False)


