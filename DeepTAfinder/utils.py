import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score, precision_recall_curve, roc_curve


def one_hot_encoding(y, num_class=2):
    return np.eye(num_class)[y]


def metrics(y, pred, score):
    accuracy = accuracy_score(y, pred)
    mcc = matthews_corrcoef(y, pred)
    f1 = f1_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    specificity = tn / (tn + fp)
    fpr, tpr, thresholds_1 = roc_curve(y, score, pos_label=1)
    precision_array, recall_array, thresholds_2 = precision_recall_curve(y, score, pos_label=1)

    # y_one_hot = one_hot_encoding(y)
    roc_auc = roc_auc_score(y, score)
    average_precision = average_precision_score(y, score)

    metrics_dict = {"Accuracy": accuracy, 
                    "Precision": precision, 
                    "Recall": recall, 
                    "Specificity": specificity, 
                    "F1-score": f1,
                    "MCC": mcc, 
                    "AUC": roc_auc, 
                    "AUPRC": average_precision,
                    'precision_array': precision_array,
                    'recall_array': recall_array,
                    'fpr': fpr,
                    'tpr': tpr,
                    'average_precision': average_precision}
                    
    return metrics_dict


def label2index(label):
    label_dict = {'Non-': 0, 'T1SE': 1, 'T2SE': 2, 'T3SE': 3, 'T4SE': 4, 'T6SE': 5}
    index = label_dict[label]
    return index


def viz_conf_matrix(cm, labels, figsize=(7, 6)):
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if p < 1:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    fig = plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    return fig
