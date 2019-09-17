map_espiras = open("mapeamento_espiras.csv")

out_map = open("create_map.csv","w")
out_map.write("latitude,longitude,name,color,note\n")

lines = map_espiras.readlines()
lines = lines[1:]

for line in lines:
	l = line.split(",")
	if l[4] == "0" or l[5]=="0":
		continue
	unique_id = l[0]+"_" +l[1]
	#l[5] = l[5].rstrip()
	out_map.write(l[5].rstrip()+"," + l[4] + "," + unique_id + "," + "#0000ff" + "," + "nothing" +"\n")


out_map.close()