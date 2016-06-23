import RankNet
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import csv
f=open("features_2.csv")
X = []
k = 0
for row in csv.reader(f):
    X.append([])
    for i in row:
        X[k].append(eval(i))
    k += 1

rankarray = [[] for i in range(len(X))]
n_att = 10

for i in range(n_att):
    print (i)
    model = 'Model_att'+str(i)+'.model'
    Model = RankNet.RankNet(model)
    X = np.asarray(X)
    py = Model.RankNetpredict(X, batchsize=10)
    for j in range(len(py)):
        rankarray[j].append(py[j])
print (rankarray)
