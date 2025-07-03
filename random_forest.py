import os 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import *
import numpy as np

# decision trees do not require standarization of data 
# standarization needed for gradient based algorithms

thumbs_up = read_csv(os.path.join(CSV_PATH,"csv_thumbs_up_landmarks.csv"))
thumbs_down = read_csv(os.path.join(CSV_PATH,"csv_thumbs_down_landmarks.csv"))
non_wanted = read_csv(os.path.join(CSV_PATH,"csv_non_wanted_landmarks.csv"))

class_thumbs_up = classify([float(item) for item in thumbs_up],0)
class_thumbs_down = classify([float(item) for item in thumbs_down],1)
class_non_wanted = classify([float(item) for item in non_wanted],2)


classes = []
classes.extend(class_thumbs_up)
classes.extend(class_thumbs_down)
classes.extend(class_non_wanted)

X = [item[0] for item in classes]
Y = [item[1] for item in classes]

arr_x = np.array(X)
arr_y = np.array(Y)
# we basically draw a series of yes/no questions and draw a flow chart
# very powerfull yet so prone to overfitting 
Xtrain , Xtest, Ytrain , Ytest = train_test_split(arr_x,arr_y,train_size=0.4,random_state=42)

tree = DecisionTreeClassifier(criterion="entropy",random_state=44,max_features="sqrt")
tree.fit(Xtrain,Ytrain)

print("tree desired values ",Ytest)
print("tree predicted values ",tree.predict(Xtest))
print("tree accuracy ",accuracy_score(Ytest,tree.predict(Xtest)))

forest = RandomForestClassifier(criterion="entropy",max_features="sqrt",)
forest.fit(Xtrain,Ytrain)
print("\n\n")
print("tree desired values ",Ytest)
print("tree predicted values ",forest.predict(Xtest))
print("tree accuracy ",accuracy_score(Ytest,forest.predict(Xtest)))

