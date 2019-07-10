#small script to merge files
import sys
import os
import pandas as pd
files = []
for i in sys.argv[1:]:
	files.append(pd.read_csv(i))
combined_csv = pd.concat(files)
#export to csv
combined_csv.to_csv("combined_csv.csv", sep= ',', index=False)


