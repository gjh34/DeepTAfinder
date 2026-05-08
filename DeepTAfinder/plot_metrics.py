import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description = "Plot metrics for fine-tuning results")
    parser.add_argument("-d", "--dir", type = str, required = True, help = "output directory")    
    parser.add_argument("-s", "--suffix", type = str, required = True, help = "suffix for output files")
    parser.add_argument("-f", "--foldnum", type = int, required = False, default = 5, help = "The number of folds in cross validation")
    parser.add_argument("-p", "--patience", type = int, required = False, help = "Patience for early stopping")
    parser.add_argument("-l", "--logdir", type = str, required = True, help = "Path to the logging directory")
    return parser.parse_args()


def get_eval_result(file, kfold, patience):
    eval_result = {}
    test_result = {}

    for i in range(1, kfold+1):
        eval_result_file = file.format(i)
        with open(eval_result_file, 'r') as f:
            lines = f.readlines()
            try:
                last_epoch = int(re.search("Early stopping at Epoch (\d+)", lines[-2]).group(1))
                best_epoch = last_epoch - patience
                eval_result_line = lines[best_epoch * 3 - 1]
            except AttributeError:
                eval_result_line = lines[-2]
            test_result_line = lines[-1]
            
            # Get validation results
            if re.search('Train Loss', eval_result_line):
                continue
            eval_result_line = eval_result_line.replace('%', '')
            eval_result_line = eval_result_line.split("INFO:")[1]            
            results = eval_result_line.split('|')
            for metric in results:
                metric = metric.strip()
                metric = metric.split(' ')
                metric[0] = metric[0].strip(':')
                if not metric[0] in eval_result.keys():
                    eval_result[metric[0]] = []

                if not metric[0] in ['fpr', 'tpr', 'precision_array', 'recall_array']:
                    eval_result[metric[0]].append(float(metric[1]))
                else:
                    array_value = metric[1:]
                    array = [float(i) for i in array_value]
                    eval_result[metric[0]].append(array)

            # Get test results
            test_result_line = test_result_line.replace('%', '')
            test_result_line = test_result_line.split("INFO:")[1]
            results = test_result_line.split('|')
            for metric in results:
                metric = metric.strip()
                metric = metric.split(' ')
                metric[0] = metric[0].strip(':')
                if not metric[0] in test_result.keys():
                    test_result[metric[0]] = []

                if not metric[0] in ['fpr', 'tpr', 'precision_array', 'recall_array']:
                    test_result[metric[0]].append(float(metric[1]))
                else:
                    array_value = metric[1:]
                    array = [float(i) for i in array_value]
                    test_result[metric[0]].append(array)
    
    return([eval_result, test_result])


def average_roc_curve_data(eval_result, kfold):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
            
    for i in range(0, kfold):
        fpr = eval_result['fpr'][i]
        tpr = eval_result['tpr'][i]
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    mean_tpr = np.mean(tprs, axis=0)
    std = np.std(tprs, axis=0)

    return([base_fpr, mean_tpr, std])


def average_precision_recall_data(eval_result, kfold):
    precisions = []
    base_recall = np.linspace(0, 1, 101)

    for i in range(0, kfold):
        recall = eval_result['recall_array'][i]
        precision = eval_result['precision_array'][i]
        # Reverse the recall_array and precision_array, so the X is in increasing order
        recall.reverse()
        precision.reverse()
        precision = np.interp(base_recall, recall, precision)
        precision[0] = 1.0
        precision = precision
        precisions.append(precision)

    mean_precision = np.mean(precisions, axis=0)
    std = np.std(precisions, axis=0)

    return([base_recall, mean_precision, std])


def plot_roc_curve(fpr, tpr, tpr_std, roc_auc, auc_std, output):
    tprs_upper = np.minimum(tpr + tpr_std, 1)
    tprs_lower = tpr - tpr_std

    plt.figure(figsize=(5, 5))
    lw = 2

    if auc_std == 0:
        label = 'ROC curve (AUC = %0.3f)' % (roc_auc)
    else:
        label = 'ROC curve (AUC = %0.3f ± %0.3f)' % (roc_auc, auc_std)

    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.fill_between(fpr, tprs_lower, tprs_upper, color='lightgray', alpha=0.4)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(linestyle='--', color='lightgray')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(output, dpi = 600, format = 'png')


def plot_precision_recall_curve(recall_array, precision_array, precision_std, average_precision, auprc_std, output):   
    precisions_upper = np.minimum(precision_array + precision_std, 1)
    precisions_lower = precision_array - precision_std
    
    plt.figure(figsize=(5, 5))
    f_scores = np.linspace(0.2, 0.8, num=4)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('F1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    if auprc_std == 0:
        label = 'Precision-recall curve (AUPRC = %0.3f)' % (average_precision)
    else:
        label = 'Precision-recall curve (AUPRC = %0.3f ± %0.3f)' % (average_precision, auprc_std)

    plt.plot(recall_array, precision_array, color='darkorange', 
                lw=2, 
                label=label)
    plt.fill_between(recall_array, precisions_lower, precisions_upper, color='lightgray', alpha=0.4)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(output, dpi = 600, format = 'png')


def plot_metrics_during_training(metrics_during_training, output):
    # Store the data
    df_metrics = pd.DataFrame(columns=['epoch', 'eval_loss', 'accuracy', 'recall', 'precision', 'specificity', 'f1', 'mcc'])

    # Define plot layout
    fig, axs = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    fit_deg = 8

    training_epoch, training_step = metrics_during_training['training_epoch'][0], metrics_during_training['training_step'][0]
    eval_epoch, eval_step = metrics_during_training['eval_epoch'][0], metrics_during_training['eval_step'][0]
    df_metrics['epoch'] = eval_epoch

    ### Learning rate
    lr, std = np.mean(metrics_during_training['learning_rate'], axis=0), np.std(metrics_during_training['learning_rate'], axis=0)
    lr_upper = np.minimum(lr + std, 1)
    lr_lower = lr - std

    axs[0, 0].plot(training_epoch, lr, color='darkorange', lw=2)
    axs[0, 0].fill_between(training_epoch, lr_lower, lr_upper, color='lightgray', alpha=0.4)
    #axs[0, 0].set_xlim([0.0, 1.05])
    #axs[0, 0].set_ylim([0.0, max(lr) * 1.05])
    axs[0, 0].grid(linestyle='--', color='lightgray')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Learning rate')
    axs[0, 0].set_title('Learning Rate')

    ### Training loss
    training_loss, std = np.mean(metrics_during_training['training_loss'], axis=0), np.std(metrics_during_training['training_loss'], axis=0)
    training_loss_upper = np.minimum(training_loss + std, 1)
    training_loss_lower = training_loss - std
    # Fit the points into curve
    #fit_func = np.poly1d(np.polyfit(training_step, training_loss, fit_deg))
    #training_loss = fit_func(training_step)

    axs[0, 1].plot(training_epoch, training_loss, color='darkorange', lw=2)
    axs[0, 1].fill_between(training_epoch, training_loss_lower, training_loss_upper, color='lightgray', alpha=0.4)
    #axs[0, 1].set_xlim([0.0, 1.05])
    #axs[0, 1].set_ylim([0.0, max(lr) * 1.05])
    axs[0, 1].grid(linestyle='--', color='lightgray')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Training loss')
    axs[0, 1].set_title('Training Loss')

    ### Validation loss
    eval_loss, std = np.mean(metrics_during_training['eval_loss'], axis=0), np.std(metrics_during_training['eval_loss'], axis=0)
    eval_loss_upper = np.minimum(eval_loss + std, 1)
    eval_loss_lower = eval_loss - std
    # Fit the points into curve
    #fit_func = np.poly1d(np.polyfit(eval_step, eval_loss, fit_deg))
    #eval_loss = fit_func(eval_step)
    df_metrics['eval_loss'] = eval_loss

    axs[0, 2].plot(eval_epoch, eval_loss, color='darkorange', lw=2)
    axs[0, 2].fill_between(eval_epoch, eval_loss_lower, eval_loss_upper, color='lightgray', alpha=0.4)
    #axs[0, 2].set_xlim([0.0, 1.05])
    #axs[0, 2].set_ylim([0.0, max(lr) * 1.05])
    axs[0, 2].grid(linestyle='--', color='lightgray')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('Validation loss')
    axs[0, 2].set_title('Validation Loss')

    ### Accuracy
    eval_acc, std = np.mean(metrics_during_training['eval_acc'], axis=0), np.std(metrics_during_training['eval_acc'], axis=0)
    eval_acc_upper = np.minimum(eval_acc + std, 1)
    eval_acc_lower = eval_acc - std
    # Fit the points into curve
    #fit_func = np.poly1d(np.polyfit(eval_step, eval_acc, fit_deg))
    #eval_acc = fit_func(eval_step)
    df_metrics['accuracy'] = eval_acc

    axs[1, 0].plot(eval_epoch, eval_acc, color='darkorange', lw=2)
    axs[1, 0].fill_between(eval_epoch, eval_acc_lower, eval_acc_upper, color='lightgray', alpha=0.4)
    #axs[1, 0].set_xlim([0.0, 1.05])
    #axs[1, 0].set_ylim([0.0, max(lr) * 1.05])
    axs[1, 0].grid(linestyle='--', color='lightgray')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].set_title('Accuracy')

    ### Recall
    eval_recall, std = np.mean(metrics_during_training['eval_recall'], axis=0), np.std(metrics_during_training['eval_recall'], axis=0)
    eval_recall_upper = np.minimum(eval_recall + std, 1)
    eval_recall_lower = eval_recall - std
    # Fit the points into curve
    #fit_func = np.poly1d(np.polyfit(eval_step, eval_recall, fit_deg))
    #eval_recall = fit_func(eval_step)
    df_metrics['recall'] = eval_recall

    axs[1, 1].plot(eval_epoch, eval_recall, color='darkorange', lw=2)
    axs[1, 1].fill_between(eval_epoch, eval_recall_lower, eval_recall_upper, color='lightgray', alpha=0.4)
    #axs[1, 1].set_xlim([0.0, 1.05])
    #axs[1, 1].set_ylim([0.0, max(lr) * 1.05])
    axs[1, 1].grid(linestyle='--', color='lightgray')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].set_title('Recall')

    ### Specificity
    eval_specificity, std = np.mean(metrics_during_training['eval_specificity'], axis=0), np.std(metrics_during_training['eval_specificity'], axis=0)
    eval_specificity_upper = np.minimum(eval_specificity + std, 1)
    eval_specificity_lower = eval_specificity - std
    # Fit the points into curve
    #fit_func = np.poly1d(np.polyfit(eval_step, eval_specificity, fit_deg))
    #eval_specificity = fit_func(eval_step)
    df_metrics['specificity'] = eval_specificity

    axs[1, 2].plot(eval_epoch, eval_specificity, color='darkorange', lw=2)
    axs[1, 2].fill_between(eval_epoch, eval_specificity_lower, eval_specificity_upper, color='lightgray', alpha=0.4)
    #axs[1, 2].set_xlim([0.0, 1.05])
    #axs[1, 2].set_ylim([0.0, max(lr) * 1.05])
    axs[1, 2].grid(linestyle='--', color='lightgray')
    axs[1, 2].set_xlabel('Epoch')
    axs[1, 2].set_ylabel('Specificity')
    axs[1, 2].set_title('Specificity')

    ### Precision
    eval_precision, std = np.mean(metrics_during_training['eval_precision'], axis=0), np.std(metrics_during_training['eval_precision'], axis=0)
    eval_precision_upper = np.minimum(eval_precision + std, 1)
    eval_precision_lower = eval_precision - std
    # Fit the points into curve
    #fit_func = np.poly1d(np.polyfit(eval_step, eval_precision, fit_deg))
    #eval_precision = fit_func(eval_step)
    df_metrics['precision'] = eval_precision

    axs[2, 0].plot(eval_epoch, eval_precision, color='darkorange', lw=2)
    axs[2, 0].fill_between(eval_epoch, eval_precision_lower, eval_precision_upper, color='lightgray', alpha=0.4)
    #axs[2, 0].set_xlim([0.0, 1.05])
    #axs[2, 0].set_ylim([0.0, max(lr) * 1.05])
    axs[2, 0].grid(linestyle='--', color='lightgray')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Precision')
    axs[2, 0].set_title('Precision')

    ### F1-score
    eval_f1, std = np.mean(metrics_during_training['eval_f1'], axis=0), np.std(metrics_during_training['eval_f1'], axis=0)
    eval_f1_upper = np.minimum(eval_f1 + std, 1)
    eval_f1_lower = eval_f1 - std
    # Fit the points into curve
    #fit_func = np.poly1d(np.polyfit(eval_step, eval_f1, fit_deg))
    #eval_f1 = fit_func(eval_step)
    df_metrics['f1'] = eval_f1

    axs[2, 1].plot(eval_epoch, eval_f1, color='darkorange', lw=2)
    axs[2, 1].fill_between(eval_epoch, eval_f1_lower, eval_f1_upper, color='lightgray', alpha=0.4)
    #axs[2, 1].set_xlim([0.0, 1.05])
    #axs[2, 1].set_ylim([0.0, max(lr) * 1.05])
    axs[2, 1].grid(linestyle='--', color='lightgray')
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('F1-score')
    axs[2, 1].set_title('F1-score')

    ### MCC
    eval_mcc, std = np.mean(metrics_during_training['eval_mcc'], axis=0), np.std(metrics_during_training['eval_mcc'], axis=0)
    eval_mcc_upper = np.minimum(eval_mcc + std, 1)
    eval_mcc_lower = eval_mcc - std
    # Fit the points into curve
    #fit_func = np.poly1d(np.polyfit(eval_step, eval_mcc, fit_deg))
    #eval_mcc = fit_func(eval_step)
    df_metrics['mcc'] = eval_mcc

    axs[2, 2].plot(eval_epoch, eval_mcc, color='darkorange', lw=2)
    axs[2, 2].fill_between(eval_epoch, eval_mcc_lower, eval_mcc_upper, color='lightgray', alpha=0.4)
    #axs[2, 2].set_xlim([0.0, 1.05])
    #axs[2, 2].set_ylim([0.0, max(lr) * 1.05])
    axs[2, 2].grid(linestyle='--', color='lightgray')
    axs[2, 2].set_xlabel('Epoch')
    axs[2, 2].set_ylabel('MCC')
    axs[2, 2].set_title('MCC')

    fig.savefig(output, dpi = 600, format = 'png')

    return df_metrics


def main():
    args = parse_args()

    kfold = args.foldnum
    output_dir = args.dir
    suffix = args.suffix
    patience = args.patience
    log_dir = args.logdir
    # Extract results and plot metrics
    eval_result, test_result = get_eval_result(os.path.join(log_dir, 'fold_{}/training.log'), kfold, args.patience)
    best_fold = np.argmax(eval_result['F1'])
    best_fold_test = np.argmax(test_result['F1'])

    # Write results
    # Validation
    metrics_acc = str(round(np.mean(eval_result['Acc']), 1)) + ' ± ' + str(round(np.std(eval_result['Acc']), 1))
    metrics_precision = str(round(np.mean(eval_result['Precision']), 1)) + ' ± ' + str(round(np.std(eval_result['Precision']), 1))
    metrics_recall = str(round(np.mean(eval_result['Recall']), 1)) + ' ± ' + str(round(np.std(eval_result['Recall']), 1))
    metrics_specificity = str(round(np.mean(eval_result['Specificity']), 1)) + ' ± ' + str(round(np.std(eval_result['Specificity']), 1))
    metrics_f1 = str(round(np.mean(eval_result['F1']), 3)) + ' ± ' + str(round(np.std(eval_result['F1']), 3))
    metrics_mcc = str(round(np.mean(eval_result['MCC']), 3)) + ' ± ' + str(round(np.std(eval_result['MCC']) ,3))
    metrics_auc = str(round(np.mean(eval_result['AUC']), 3)) + ' ± ' + str(round(np.std(eval_result['AUC']) ,3))
    metrics_auprc = str(round(np.mean(eval_result['average_precision']), 3)) + ' ± ' + str(round(np.std(eval_result['average_precision']) ,3))

    # Test
    metrics_acc_test = str(round(np.mean(test_result['Acc']), 1)) + ' ± ' + str(round(np.std(test_result['Acc']), 1))
    metrics_precision_test = str(round(np.mean(test_result['Precision']), 1)) + ' ± ' + str(round(np.std(test_result['Precision']), 1))
    metrics_recall_test = str(round(np.mean(test_result['Recall']), 1)) + ' ± ' + str(round(np.std(test_result['Recall']), 1))
    metrics_specificity_test = str(round(np.mean(test_result['Specificity']), 1)) + ' ± ' + str(round(np.std(test_result['Specificity']), 1))
    metrics_f1_test = str(round(np.mean(test_result['F1']), 3)) + ' ± ' + str(round(np.std(test_result['F1']), 3))
    metrics_mcc_test = str(round(np.mean(test_result['MCC']), 3)) + ' ± ' + str(round(np.std(test_result['MCC']) ,3))
    metrics_auc_test = str(round(np.mean(test_result['AUC']), 3)) + ' ± ' + str(round(np.std(test_result['AUC']) ,3))
    metrics_auprc_test = str(round(np.mean(test_result['average_precision']), 3)) + ' ± ' + str(round(np.std(test_result['average_precision']) ,3))

    with open(output_dir + '/eval_result_' + suffix + '.txt', 'w') as f:
        f.write('Validation result:\n \
            accuracy: {}%\n \
            recall: {}%\n \
            precision: {}%\n \
            specificity: {}%\n \
            f1: {}\n \
            mcc: {}\n \
            auc: {}\n \
            auprc: {}\n\n'.format(
                metrics_acc,
                metrics_recall,
                metrics_precision,
                metrics_specificity,
                metrics_f1,
                metrics_mcc,
                metrics_auc,
                metrics_auprc
        ))
        f.write('Test result:\n \
            accuracy: {}%\n \
            recall: {}%\n \
            precision: {}%\n \
            specificity: {}%\n \
            f1: {}\n \
            mcc: {}\n \
            auc: {}\n \
            auprc: {}\n\n'.format(
                metrics_acc_test,
                metrics_recall_test,
                metrics_precision_test,
                metrics_specificity_test,
                metrics_f1_test,
                metrics_mcc_test,
                metrics_auc_test,
                metrics_auprc_test
        ))
        f.write('Best fold in validation: {}\n \
            accuracy: {}%\n \
            recall: {}%\n \
            precision: {}%\n \
            specificity: {}%\n \
            f1: {}\n \
            mcc: {}\n \
            auc: {}\n \
            auprc: {}\n'.format(
                best_fold + 1,
                str(eval_result['Acc'][best_fold]),
                str(eval_result['Recall'][best_fold]),
                str(eval_result['Precision'][best_fold]),
                str(eval_result['Specificity'][best_fold]),
                str(eval_result['F1'][best_fold]),
                str(eval_result['MCC'][best_fold]),
                str(eval_result['AUC'][best_fold]),
                str(eval_result['average_precision'][best_fold]),
        ))
        f.write('Best fold in test: {}\n \
            accuracy: {}%\n \
            recall: {}%\n \
            precision: {}%\n \
            specificity: {}%\n \
            f1: {}\n \
            mcc: {}\n \
            auc: {}\n \
            auprc: {}\n'.format(
                best_fold_test + 1,
                str(test_result['Acc'][best_fold_test]),
                str(test_result['Recall'][best_fold_test]),
                str(test_result['Precision'][best_fold_test]),
                str(test_result['Specificity'][best_fold_test]),
                str(test_result['F1'][best_fold_test]),
                str(test_result['MCC'][best_fold_test]),
                str(test_result['AUC'][best_fold_test]),
                str(test_result['average_precision'][best_fold_test]),
        ))
        
    # ROC curve
    fpr, tpr, tpr_std = average_roc_curve_data(eval_result, kfold)
    roc_curve_output = output_dir + "/ROC_curve_{" + suffix + "}.png"
    #plot_roc_curve(fpr, tpr, tpr_std, eval_result['AUC'][best_fold], roc_curve_output)
    plot_roc_curve(fpr, tpr, tpr_std, np.mean(eval_result['AUC']), np.std(eval_result['AUC']), roc_curve_output)
    
    # Precision-recall curve
    recall, precision, precision_std = average_precision_recall_data(eval_result, kfold)
    precision_recall_curve_output = output_dir + "/precision_recall_curve_{" + suffix + "}.png"
    #plot_precision_recall_curve(recall, precision, precision_std, eval_result['average_precision'][best_fold], precision_recall_curve_output)
    plot_precision_recall_curve(recall, precision, precision_std, np.mean(eval_result['average_precision']), np.std(eval_result['average_precision']), precision_recall_curve_output)


if __name__ == '__main__':
    main()

