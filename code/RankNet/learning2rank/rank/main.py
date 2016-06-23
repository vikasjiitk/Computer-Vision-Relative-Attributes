import RankNet
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# feat_file = 'feat.txt'
# # X = np.loadtxt(feat_file)
# f1 = open(feat_file, 'r')
# a = f1.read().split('\n')
# X = []

# for i in range(len(a)):
#     feature = a[i].split("\t")
#     X.append(feature[0:512])
import csv
f=open("features.csv")
X = []
k = 0
for row in csv.reader(f):
    # if k>=100:
    #     break
    X.append([])
    for i in row:
        X[k].append(eval(i))
    k += 1

label_file = 'labels.txt'
f2 = open(label_file,'r')
l = f2.read().split('\n')
# l = l[0:100]
# print (len(X),len(l))
freq = []
for i in range(1,9):
    freq.append(l.count(str(i)))

# print (freq)
X_tr = []
l_tr = []
X_ts = []
l_ts = []

index = 0
for p in freq:
    i = index
    j = index + 85
    k = index + p

    X_tr.extend(X[i:j])
    l_tr.extend(l[i:j])
    X_ts.extend(X[j:k])
    l_ts.extend(l[j:k])

    index += p

# print len(X_tr)
# for i in range(len(X_ts)):
#     for j in range(len(X_ts[i])):
#         print X_ts[i][j],
#         if(j!=len(X_ts[i])-1):
#             print(', '),
#     print '\n'

class_strength_main = [[6, 8, 7, 5, 2, 1, 4, 3],
 [1, 2, 3, 5, 7, 6, 8, 4],
 [5, 3, 2, 4, 8, 6, 1, 7],
 [8, 4, 3, 2, 6, 7, 1, 5],
 [5, 6, 7, 1, 3, 4, 8, 2],
 [6, 7, 5, 8, 1, 2, 4, 3],
 [4, 6, 5, 2, 1, 3, 7, 8],
 [1, 2, 8, 7, 4, 6, 5, 3],
 [7, 5, 1, 2, 6, 8, 3, 4],
 [6, 4, 1, 3, 8, 7, 2, 5]]

rankarray = [[] for i in range(len(X_tr)+len(X_ts) ) ]

for i in range(len(class_strength_main)):
    class_strength=class_strength_main[i]
    y_tr=[]
    y_ts = []
    for j in range(len(l_tr)):
        y_tr.append(class_strength[int(l_tr[j])-1])
        # print (y_tr[j])

    for j in range(len(l_ts)):
        y_ts.append(class_strength[int(l_ts[j])-1])
        # print (y_ts[j])

    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr)
    X_ts = np.array(X_ts)
    y_ts = np.array(y_ts)
    # print (len(y_tr),len(X_tr))
    # y = np.concatenate((y_tr,y_ts), axis=0)
    Model = RankNet.RankNet()

    Model.fit(np.concatenate((X_tr,X_ts), axis=0),np.concatenate((y_tr,y_ts), axis=0),batchsize=10, n_iter=10000, n_units1 = 2048)


    py = Model.RankNetpredict(np.concatenate((X_tr,X_ts), axis=0),batchsize=10)
    for j in range(len(py)):
        rankarray[j].append(py[j])
        # print (py[j],y[j])
    # print ('hello')
    Model.saveModels('Model_att'+str(i)+'.model')
print (rankarray)
