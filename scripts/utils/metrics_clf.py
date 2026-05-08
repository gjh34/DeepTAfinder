"""
Metrics for protein sequence classification tasks.
Based on GLUE convention.
"""
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, recall_score, precision_score, roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        'acc': acc,
        'f1': f1,
        'acc_and_f1': (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        'pearson': pearson_corr,
        'spearmanr': spearman_corr,
        'corr': (pearson_corr + spearman_corr) / 2,
    }


def glue_compute_metrics(task_name, preds, labels, scores = []):
    assert len(preds) == len(labels)
    if task_name == 'cola':
        return {'mcc': matthews_corrcoef(labels, preds)}
    elif task_name == 'sst-2':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'solubility':
        return {
            'mcc': matthews_corrcoef(preds, labels), 
            'acc': simple_accuracy(preds, labels)
            }
    elif task_name == 'localization':
        return {
            'mcc': matthews_corrcoef(preds, labels), 
            'acc': simple_accuracy(preds, labels)
            }
    elif task_name == 'remote-homology':
        return {
            'mcc': matthews_corrcoef(preds, labels), 
            'acc': simple_accuracy(preds, labels)
            }
    elif task_name == 'pairwise-interaction':
        return acc_and_f1(preds, labels)
    elif task_name == 'pairwise-string':
        return acc_and_f1(preds, labels)
    elif task_name == 'pairwise-ta':
        fpr, tpr, thresholds_1 = roc_curve(y_true=labels, y_score=scores, pos_label=1)
        tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
        precision, recall, thresholds_2 = precision_recall_curve(y_true=labels, probas_pred=scores, pos_label=1)

        return {
            'acc': accuracy_score(y_true=labels, y_pred=preds),
            'mcc': matthews_corrcoef(y_true=labels, y_pred=preds),
            'f1': f1_score(y_true=labels, y_pred=preds),
            'recall': recall_score(y_true=labels, y_pred=preds),
            'precision': precision_score(y_true=labels, y_pred=preds),
            'specificity': tn / (tn + fp),
            'fpr': fpr,
            'tpr': tpr,
            'fn': fn,
            'tn': tn,
            'fp': fp,
            'tp': tp,
            'roc_auc': roc_auc_score(y_true=labels, y_score=scores),
            'precision_array': precision,
            'recall_array': recall,
            'average_precision': average_precision_score(y_true=labels, y_score=scores),
            #'roc_thresholds': thresholds_1,
            #'precision_recall_thresholds': thresholds_2            
        }
    elif task_name == 'mrpc':
        return acc_and_f1(preds, labels)
    elif task_name == 'sts-b':
        return pearson_and_spearman(preds, labels)
    elif task_name == 'qqp':
        return acc_and_f1(preds, labels)
    elif task_name == 'mnli':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'mnli-mm':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'qnli':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'rte':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'wnli':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'hans':
        return {'acc': simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def plot_roc_curve(fpr, tpr, roc_auc, output):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(output, dpi = 600, format = 'png')