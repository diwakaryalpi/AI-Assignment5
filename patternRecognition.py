import pickle
import numpy as np
from sklearn import preprocessing
from random import shuffle
from sklearn.metrics._classification import classification_report
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier

f = open('datafile.txt','r')
flag=0
lines=f.readlines()
length = len(lines)
X = []
Y = []
temp=[]
count=0
for i in range(length):
    if count<32:
        sp = list(map(int,lines[i][:-1]))
        temp.extend(sp)
        count+=1
    elif count==32:
        X.append(temp)
        Y.append(str(lines[i][:-1]))
        temp=[]
        count=0
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1000), random_state=1)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
target_names = ['0','1', '2','3','4','5','6','7','8','9']

f1 = open('finalresult.txt','w')
count=0
for i in range(len(X_test)):
    tempStr = ''.join(str(v) for v in X_test[i])
    finalStr = ""
    for j in range(32):
        finalStr += tempStr[j*32:j*32+31]
        finalStr+="\n"

    finalStr = finalStr + "\n"+ str(y_test[i])+ " " +str(y_predict[i]) +"\n"
    if y_test[i] != y_predict[i]:
        count+=1
        f1.write(finalStr)

print(classification_report(y_test, y_predict, target_names=target_names))
print("Unmatched cases: ", count," out of ", len(X_test), " instances")
f.close()
f1.close()
