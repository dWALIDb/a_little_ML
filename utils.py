import csv


CSV_PATH = r".\dataset"


def read_csv(csv_path:str)->list:
    values=[]
    csv_file = open(csv_path,mode="r",newline='')
    csv_reader=csv.reader(csv_file)
    for i,row in enumerate(csv_reader):
        if i==0 : continue #the keys are in the first line
        values.insert(i-1,row[1]) #the values on the next lines :)
   
    return values

def classify(data:list,_class:int)->list:
    #we have 20 landmarks(wrist is not considered)
    #we create a list that has 20 data points per class ... 
    l=[]
    result=[]
    for it in range(len(data)):
        if it % 20 == 0 and it != 0 :
            # gotta copy the elements to the new list else you would get
            # elemenst of the last iteration only... repeated like 70 times 
            result.append([l.copy(),_class])
            l.clear()
        l.append(data[it])

    return result
