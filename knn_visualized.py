import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#file that has some utilities like reading csv, and classifiying the read data :)
from utils import *

# used to change n_neighbors parameter
MAX_NEIGHBOURS=20
# read csv files and put in dictionaries 

thumbs_up=read_csv(os.path.join(CSV_PATH,"csv_thumbs_up_landmarks.csv"))
thumbs_down=read_csv(os.path.join(CSV_PATH,"csv_thumbs_down_landmarks.csv"))
non_wanted=read_csv(os.path.join(CSV_PATH,"csv_non_wanted_landmarks.csv"))


classes = []
# list comprehension and classifiying the known data points :)
class_thumbs_up=classify([float(item) for item in thumbs_up],0)
class_thumbs_down=classify([float(item) for item in thumbs_down],1)
class_non_wanted=classify([float(item) for item in non_wanted],2)
# the lisits have the data points and the corresponding class for each element
# thumbs up class 0 , thumbs down 1 , non wanted 2 :) 
classes.extend(class_thumbs_up)
classes.extend(class_thumbs_down)
classes.extend(class_non_wanted)



# i got 61 elements
X= [item[0] for item in classes]
Y= [item[1] for item in classes]

arr_x=np.array(X)
arr_y=np.array(Y)

accuracies_train=[]
accuracies_test=[]
scaled_accuracies_train=[]
scaled_accuracies_test=[]

Xtrain,Xtest,Ytrain,Ytest=train_test_split(arr_x,arr_y,test_size=0.3,random_state=42)
# testing the standarization of features
scaler = StandardScaler()
scaled_train = scaler.fit_transform(Xtrain)
scaled_test = scaler.transform(Xtest)

for i in range(1,MAX_NEIGHBOURS,1):
    # cosine metric does the best so far 
    # even though it is a similarity measure, used in test analysis and image retrieval
    knn=KNeighborsClassifier(n_neighbors=i,weights='distance',metric= 'cosine')
    knn_scaled=KNeighborsClassifier(n_neighbors=i,weights='distance',metric="cosine")
    knn.fit(Xtrain,Ytrain)
    knn_scaled.fit(scaled_train,Ytrain)
    accuracies_test.append(accuracy_score(Ytest,knn.predict(Xtest)))
    accuracies_train.append(accuracy_score(Ytrain,knn.predict(Xtrain)))
    scaled_accuracies_test.append(accuracy_score(Ytest,knn_scaled.predict(scaled_test)))
    scaled_accuracies_train.append(accuracy_score(Ytrain,knn_scaled.predict(scaled_train)))

width=np.arange(1,MAX_NEIGHBOURS,1)
plt.plot(width,np.array(accuracies_test) * 100,'-g')
plt.plot(width,np.array(accuracies_train) * 100,'-b')
plt.plot(width,np.array(scaled_accuracies_test) * 100,'-y')
plt.plot(width,np.array(scaled_accuracies_train) * 100,'-r')
plt.axis((0,MAX_NEIGHBOURS,0,100))
plt.legend(["testing accuracy","training accuracy","testing accuracy scaled","training accuracy scaled"])
plt.title("Change of Accuracy with Neibours Parameter and Standarization")
plt.xlabel("N_neibours")
plt.ylabel("Accuracy")
plt.show()