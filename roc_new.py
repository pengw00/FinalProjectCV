import pylab as pl
import numpy as np
import numpy.random as r
import sklearn.metrics as m
import matplotlib.pyplot as plt
from math import log,exp,sqrt
result="sent333.txt"
db = [] #[score,nonclk,clk]
pos, neg = 0, 0
fs = open(result,'r').read().strip().split('\n')
for line in fs:
    nonclk,clk,score = line.split('\t')
    nonclk = int(nonclk)
    clk = int(clk)
    score = float(score)
    #print([score,nonclk,clk])
    db.append([score,nonclk,clk])
    pos += clk
    neg += nonclk
#db = sorted(db, key=lambda x:x[0], reverse=True)
#calculate ROC position
#print(db)

xy_arr = []
tp, fp = 0., 0.
for i in range(len(db)):
 	tp += db[i][2]
 	fp += db[i][1]
 	xy_arr.append([fp/neg,tp/pos])
auc = 0.
prev_x = 0
for x,y in xy_arr:
 	if x != prev_x:
 		auc += (x - prev_x) * y
 		prev_x = x
#print("the auc is %s."%auc)
x = [_v[0] for _v in xy_arr]
y = [_v[1] for _v in xy_arr]
pl.title("ROC curve of %s (AUC = %.4f)" % ('svm',auc))
pl.xlabel("False Positive Rate")
pl.ylabel("True Positive Rate")
pl.plot(x, y)# use pylab to plot x and y
pl.show()# show the plot on the screen
# I make a adjust the code for easy to understand
for item in db:
    print(type(item[0]))

#show precision, with y_pre is the true labels,  y_pre the predict label will be  the score < 0.2.
#print(db)
y_pre = np.array([ 1 if item[0] > float(0.3) else 0 for item in db ], dtype=np.float32)
y_true = np.array([ 1 if item[1] >= 0.5 else 0 for item in db ], dtype=np.float32)
print(y_true)
fpr, tpr, th = m.roc_curve(y_true, y_pre)
#ax = pl.subplot(2, 1, 1)
#ax.plot(fpr, tpr)
#ax.set_title('ROC curve')

precision, recall, th = m.precision_recall_curve(y_true, y_pre)
pl.title('Precision recall curve')
pl.plot(recall, precision)
pl.ylabel('Precision')
pl.xlabel('recall')


pl.show()

num_list = [1.5, 0.6, 7.8, 6]
plt.bar(range(len(num_list)), num_list)
plt.show()
