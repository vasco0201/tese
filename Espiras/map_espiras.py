map_espiras = open("mapeamento_espiras.csv")

out_map = open("create_map.txt","w")

lines = map_espiras.readlines()
lines = lines[1:]

for line in lines:
	l = line.split(",")
	if l[4] == "0" or l[5]=="0":
		continue
	#l[5] = l[5].rstrip()
	out_map.write(l[5].rstrip()+"," + l[4] + "," + l[3] + "," + "#0000ff" + "\n")


