import numpy as np
import numpy.random as r
import sklearn.metrics as m
import pylab as pl

def main():
    size = 1000000
    y_true = np.array([ 1 if i >= 0.3 else 0 for i in r.random(size) ], dtype=np.float32)
    y_pred = r.random(size)
    y_cls = np.array([ 1 if i >= 0.5 else 0 for i in y_pred ], dtype=np.float32)

    print(m.classification_report(y_true, y_cls))

    fpr, tpr, th = m.roc_curve(y_true, y_pred)
    ax = pl.subplot(2, 1, 1)
    ax.plot(fpr, tpr)
    ax.set_title('ROC curve')

    precision, recall, th = m.precision_recall_curve(y_true, y_pred)
    ax = pl.subplot(2, 1, 2)
    ax.plot(recall, precision)
    ax.set_ylim([0.0, 1.0])
    ax.set_title('Precision recall curve')

    pl.show()


if __name__ == '__main__':
    main()