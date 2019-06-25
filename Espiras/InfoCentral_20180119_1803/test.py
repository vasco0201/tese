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

h = get_hours()
print(len(h))
header =["Data","Zona","Contadores","ID_Espira","Nr_carros"] + h
print (header)