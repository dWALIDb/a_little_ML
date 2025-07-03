from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os

thumbs_up =read_csv(os.path.join(CSV_PATH,"csv_thumbs_up_landmarks.csv"))
thumbs_down =read_csv(os.path.join(CSV_PATH,"csv_thumbs_down_landmarks.csv"))
non_wanted =read_csv(os.path.join(CSV_PATH,"csv_non_wanted_landmarks.csv"))

classes=[]
class_thumbs_up = classify([float(item)for item in thumbs_up],0)
class_thumbs_down = classify([float(item)for item in thumbs_down],1)
class_non_wanted = classify([float(item)for item in non_wanted],2)

classes.extend(class_thumbs_up)
classes.extend(class_thumbs_down)
classes.extend(class_non_wanted)

X = [item[0] for item in classes]
Y = [item[1] for item in classes]

arr_x = np.array(X)
arr_y = np.array(Y)

# we scale the values such that they have mean=0 and variation of 1 :)
scaler = StandardScaler()
Xtrain , Xtest , Ytrain , Ytest = train_test_split(arr_x,arr_y,test_size=0.2,random_state=42)
Xtrain_scaled = scaler.fit_transform(Xtrain) #get the mean of the training data
Xtest_scaled = scaler.transform(Xtest) # use the values to scale testing data

# 1000 epochs with auto batch size :)
mlp = MLPClassifier(max_iter=1000,solver="adam",hidden_layer_sizes=(7,6),random_state=42,learning_rate="adaptive")
mlp_nonscaled = MLPClassifier(max_iter=1000,solver="adam",hidden_layer_sizes=(7,6),random_state=42,learning_rate="adaptive")

# size of (7,6) and adam optimizer have the best results so far 
mlp.fit(Xtrain_scaled,Ytrain)
mlp_nonscaled.fit(Xtrain,Ytrain)

# we can notice that only the traning accuracy changes on the outputs
print(f"ANN accuracy of training {accuracy_score(Ytrain,mlp.predict(Xtrain_scaled))}")
print(f"ANN accuracy of testing {accuracy_score(Ytest,mlp.predict(Xtest_scaled))}")
print(f"ANN accuracy of training non scaled {accuracy_score(Ytrain,mlp_nonscaled.predict(Xtrain))}")
print(f"ANN accuracy of testing non scaled {accuracy_score(Ytest,mlp_nonscaled.predict(Xtest))}")


print("ANN testing first point scaled ",mlp.predict(Xtest_scaled),".")
print("ANN testing first point non scaled ",mlp_nonscaled.predict(Xtest),'.')
print("ANN actual class",Ytest,'.\n\n')

sv = SVC()
sv.fit(Xtrain_scaled,Ytrain)

sv_nonscaled = SVC(kernel="sigmoid")
sv_nonscaled.fit(Xtest,Ytest)

print(f"SVM accuracy of training {accuracy_score(Ytrain,sv.predict(Xtrain_scaled))}")
print(f"SVM accuracy of testing {accuracy_score(Ytest,sv.predict(Xtest_scaled))}")
print(f"SVM accuracy of training non scaled {accuracy_score(Ytrain,sv_nonscaled.predict(Xtrain))}")
print(f"SVM accuracy of testing non scaled {accuracy_score(Ytest,sv_nonscaled.predict(Xtest))}")


print("SVM testing first point scaled ",sv.predict(Xtest_scaled),".")
print("SVM testing first point non scaled ",sv_nonscaled.predict(Xtest),'.')
print("SVM actual class",Ytest,'.')