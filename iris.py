# CSV contains iris data as follows 
# Sepal Lenght, Sepal Width, Petal Length, Petal Width
# All in cm
# Column 5 is classification (3 total classes)

from sklearn import tree
import pandas as pd 

dataframe = pd.read_csv('iris.csv', names=['sep length', 'sep width', 'ped length', 'ped width', 'class'])
MeasDF = dataframe[['sep length', 'sep width', 'ped length', 'ped width']].copy()
ClassDF = dataframe[['class']].copy()

measlist = MeasDF.values.tolist()
classlist = ClassDF.values.tolist()

clf = tree.DecisionTreeClassifier()

clf = clf.fit(measlist,classlist)

prediction = clf.predict([[5.0, 3.4, 2.9, .8]])
print(prediction)

#TODO(brody): Should come back and make it general, i.e. ask for user input, user inputs all 4 parameters, return the prediction.