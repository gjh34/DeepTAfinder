import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import re
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description = "Plot metrics for fine-tuning results")
    parser.add_argument("-d", "--dir", type = str, required = True, help = "output directory")    
    parser.add_argument("-s", "--suffix", type = str, required = True, help = "suffix for output files")
    parser.add_argument("-c", "--nocrossvalidation", action = "store_true", required = False, default = False, help = "Results are not from cross validation, default False")
    parser.add_argument("-f", "--foldnum", type = int, required = False, default = 5, help = "The number of folds in cross validation")
    parser.add_argument("-p", "--patience", type = int, required = False, help = "Patience for early stopping")
    parser.add_argument("-l", "--logdir", type = str, required = True, help = "Path to the logging directory")
    return parser.parse_args()


def get_eval_result(cv, dir, kfold, patience):
    eval_result = {}
    test_result = {}

    for i in range(1, kfold+1):
        eval_result_file = os.path.join(dir, 'fold_{}'.format(i), 'eval_results.json')
        test_result_file = os.path.join(dir, 'fold_{}'.format(i), 'test_results.json')
        with open(eval_result_file, 'r') as json_file:
            results = json.load(json_file)
            for metric, value in results.items():
                if not metric in eval_result.keys():
                    eval_result[metric] = []
                if not metric in ['fpr', 'tpr', 'precision_array', 'recall_array']:
                    eval_result[metric].append(float(value))
                else:
                    array = [float(i) for i in value]
                    eval_result[metric].append(array)

        with open(test_result_file, 'r') as json_file:
            results = json.load(json_file)
            for metric, value in results.items():
                if not metric in test_result.keys():
                    test_result[metric] = []
                if not metric in ['fpr', 'tpr', 'precision_array', 'recall_array']:
                    test_result[metric].append(float(value))
                else:
                    array = [float(i) for i in value]
                    test_result[metric].append(array)
    
    return([eval_result, test_result])


def main():
    args = parse_args()

    cv = args.nocrossvalidation
    kfold = args.foldnum
    output_dir = args.dir
    suffix = args.suffix
    patience = args.patience
    log_dir = args.logdir
    # Extract results and plot metrics
    eval_result, test_result = get_eval_result(cv, log_dir, kfold, args.patience)

    best_fold = np.argmax(eval_result['F1-score'])
    best_fold_test = np.argmax(test_result['F1-score'])

    # Write results
    # Validation
    metrics_acc = str(round(np.mean(eval_result['Accuracy']) * 100, 1)) + ' ± ' + str(round(np.std(eval_result['Accuracy']) * 100, 1))
    metrics_precision = str(round(np.mean(eval_result['Precision']) * 100, 1)) + ' ± ' + str(round(np.std(eval_result['Precision']) * 100, 1))
    metrics_recall = str(round(np.mean(eval_result['Recall']) * 100, 1)) + ' ± ' + str(round(np.std(eval_result['Recall']) * 100, 1))
    metrics_specificity = str(round(np.mean(eval_result['Specificity']) * 100, 1)) + ' ± ' + str(round(np.std(eval_result['Specificity']) * 100, 1))
    metrics_f1 = str(round(np.mean(eval_result['F1-score']), 3)) + ' ± ' + str(round(np.std(eval_result['F1-score']), 3))
    metrics_mcc = str(round(np.mean(eval_result['MCC']), 3)) + ' ± ' + str(round(np.std(eval_result['MCC']) ,3))
    metrics_auc = str(round(np.mean(eval_result['AUC']), 3)) + ' ± ' + str(round(np.std(eval_result['AUC']) ,3))
    metrics_auprc = str(round(np.mean(eval_result['average_precision']), 3)) + ' ± ' + str(round(np.std(eval_result['average_precision']) ,3))

    # Test
    metrics_acc_test = str(round(np.mean(test_result['Accuracy']) * 100, 1)) + ' ± ' + str(round(np.std(test_result['Accuracy']) * 100, 1))
    metrics_precision_test = str(round(np.mean(test_result['Precision']) * 100, 1)) + ' ± ' + str(round(np.std(test_result['Precision']) * 100, 1))
    metrics_recall_test = str(round(np.mean(test_result['Recall']) * 100, 1)) + ' ± ' + str(round(np.std(test_result['Recall']) * 100, 1))
    metrics_specificity_test = str(round(np.mean(test_result['Specificity']) * 100, 1)) + ' ± ' + str(round(np.std(test_result['Specificity']) * 100, 1))
    metrics_f1_test = str(round(np.mean(test_result['F1-score']), 3)) + ' ± ' + str(round(np.std(test_result['F1-score']), 3))
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
                str(eval_result['Accuracy'][best_fold] * 100),
                str(eval_result['Recall'][best_fold] * 100),
                str(eval_result['Precision'][best_fold] * 100),
                str(eval_result['Specificity'][best_fold] * 100),
                str(eval_result['F1-score'][best_fold]),
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
                str(test_result['Accuracy'][best_fold_test] * 100),
                str(test_result['Recall'][best_fold_test] * 100),
                str(test_result['Precision'][best_fold_test] * 100),
                str(test_result['Specificity'][best_fold_test] * 100),
                str(test_result['F1-score'][best_fold_test]),
                str(test_result['MCC'][best_fold_test]),
                str(test_result['AUC'][best_fold_test]),
                str(test_result['average_precision'][best_fold_test]),
        ))


if __name__ == '__main__':
    main()

