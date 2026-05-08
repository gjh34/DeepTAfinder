import os
import time
import logging
import sys
import json
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lrs
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler

from DeepTAfinder.model import ESM2Model, ESM2ModelFinetuneNotShareParams, ESM2XGBoostClassifier
from DeepTAfinder.dataset import TASequenceDataSet, Alphabet
from DeepTAfinder.utils import label2index, viz_conf_matrix
from DeepTAfinder.trainer import train, test, set_seed, EarlyStopping
from DeepTAfinder.plot_embedding import plot_embedding, get_embedding
from DeepTAfinder.losses import BCEFocalLoss

def main(args):

    set_seed(args.seed)

    # Configure logging
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(log_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)
    #writer = SummaryWriter(log_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    # If use GAN
    batch_size = args.batch_size
    if args.gan:
        batch_size = batch_size - args.gan_batch_size

    # Configure model
    if args.model == "esm2model":
        model = ESM2Model(1280, repr_layer=args.repr_layers, num_layers=args.num_layers, hid_dim=args.hid_dim, dropout_rate=args.dropout_rate, num_classes=1, return_attn=args.return_attn)
    elif args.model == "esm2model_simple":
        model = ESM2ModelSimple(1280, repr_layer=args.repr_layers, num_layers=args.num_layers, hid_dim=args.hid_dim, dropout_rate=args.dropout_rate, num_classes=1)
    elif args.model == "esm2model_finetune_notshareparams":
        model = ESM2ModelFinetuneNotShareParams(1280, repr_layer=args.repr_layers, num_layers=args.num_layers, hid_dim=args.hid_dim, dropout_rate=args.dropout_rate, num_classes=1)
    elif args.model == "esm2model_XGBoost":
        model = ESM2XGBoostClassifier(1280, repr_layer=args.repr_layers, device=device)
    else:
        raise ValueError('Invalid model type!')

    if not args.model == "esm2model_XGBoost":
        model.to(device)

    # Configure datasets and dataloaders
    alphabet = Alphabet.from_architecture("roberta_large")
    if args.mode == 'train':
        train_dataset = TASequenceDataSet(file_path=os.path.join(args.data_dir, 'train.tsv'), seed=args.seed, batch_size=batch_size, shuffle=False, stratified=args.stratified, stratified_TA_non_TA=args.stratified_TA_non_TA)
        valid_dataset = TASequenceDataSet(file_path=os.path.join(args.data_dir, 'dev.tsv'), seed=args.seed, batch_size=batch_size, shuffle=False, stratified=args.stratified)
        test_dataset = TASequenceDataSet(file_path=os.path.join(args.data_dir, 'test.tsv'), seed=args.seed, batch_size=batch_size, shuffle=False, stratified=args.stratified)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers)
    if args.mode == 'test' or args.mode == 'predict' or args.plot_embedding:
        test_dataset = TASequenceDataSet(file_path=os.path.join(args.data_dir, 'test.tsv'), seed=args.seed, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers)

    # Configure loss, optimizer, scheduler, early stopping
    #pos_num = list(train_dataset.sequence_labels).count(1)
    #neg_num = list(train_dataset.sequence_labels).count(0)
    #CE_weights = [1 / neg_num, 1 / pos_num]
    #CE_weights = torch.FloatTensor(CE_weights).to(device)
    #criterion = nn.CrossEntropyLoss(weight=CE_weights)
    #criterion = nn.CrossEntropyLoss()
    criterion = BCEFocalLoss()
    #criterion = SupConLoss()
    
    if args.mode == 'train':
        if args.model == "esm2model":
            optimizer_settings = [{
                'params':  filter(lambda p: p.requires_grad, model.pretrained_model.parameters()), 'lr': args.enc_lr
            }, {
                'params':  filter(lambda p: p.requires_grad, model.clf.parameters()), 'lr': args.lr
            }, {
                'params':  filter(lambda p: p.requires_grad, model.transformer_layer_1.parameters()), 'lr': args.enc_lr
            }, {
                'params':  filter(lambda p: p.requires_grad, model.transformer_layer_2.parameters()), 'lr': args.enc_lr
            }, {
                'params':  filter(lambda p: p.requires_grad, model.conv_1.parameters()), 'lr': args.enc_lr
            }, {
                'params':  filter(lambda p: p.requires_grad, model.conv_2.parameters()), 'lr': args.enc_lr
            }]
            optimizer = optim.Adam(optimizer_settings, weight_decay=args.weight_decay)
        elif args.model == "esm2model_finetune_notshareparams":
            optimizer_settings = [{
                'params':  filter(lambda p: p.requires_grad, model.pretrained_model_1.parameters()), 'lr': args.enc_lr
            }, {
                'params':  filter(lambda p: p.requires_grad, model.pretrained_model_2.parameters()), 'lr': args.enc_lr
            }, {
                'params':  filter(lambda p: p.requires_grad, model.clf.parameters()), 'lr': args.lr
            }]
            optimizer = optim.Adam(optimizer_settings, weight_decay=args.weight_decay)
        elif not args.model == "esm2model_XGBoost":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if not args.model == "esm2model_XGBoost":
            if args.lr_scheduler is None:
                scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warm_epochs)
                return [optimizer], [scheduler]
            else:
                if args.lr_scheduler == 'step':
                    after_scheduler = lrs.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
                elif args.lr_scheduler == 'cosine':
                    after_scheduler = lrs.CosineAnnealingLR(optimizer, T_max=args.lr_decay_steps, eta_min=args.lr_decay_min_lr)
                elif args.lr_scheduler == 'exp':
                    after_scheduler = lrs.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
                else:
                    raise ValueError('Invalid lr_scheduler type!')
                scheduler = GradualWarmupScheduler(
                    optimizer, multiplier=1, total_epoch=args.warm_epochs, after_scheduler=after_scheduler)

            early_stopping = EarlyStopping(
                patience=args.patience, checkpoint_dir=log_dir)

    # Extract embeddings
    if args.get_embedding:
        #pos_dataset = TASequenceDataSet(file_path=os.path.join(args.data_dir, 'pos_sample.tsv'), batch_size=args.batch_size, seed=args.seed, shuffle=False)
        pos_dataset = TASequenceDataSet(file_path=os.path.join(args.data_dir, 'test.tsv'), batch_size=args.batch_size, seed=args.seed, shuffle=False)
        pos_loader = DataLoader(pos_dataset, batch_size=args.batch_size,
                             collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers)
        
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'checkpoint.pt')))
        model.return_embedding = True
        model.keep_seq_len = args.keep_seq_len
        model.to(device)

        embeddings = get_embedding(model, pos_loader, device)
        
        if len(embeddings.shape) < 3:
            np.savetxt(args.embedding_file, embeddings)
        else:
            np.save(args.embedding_file, embeddings)
        sys.exit()

    # Plot embeddings
    if args.plot_embedding:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'checkpoint.pt')))
        model.return_embedding = True
        model.keep_seq_len = args.keep_seq_len        
        
        plot_embedding(model, test_loader, args.log_dir, device, args.data_dir, GAN=False, keep_seq_len=args.keep_seq_len)

        sys.exit()

    # Training epochs
    if args.mode == 'train':
        # Train XGboost
        if args.model == 'esm2model_XGBoost':
            xgb_model = model.train(train_loader, valid_loader)
            eval_results = model.evaluate(valid_loader)
            test_results = model.evaluate(test_loader)

            # Save model
            model.save_model(os.path.join(log_dir, "esm2_xgboost_model.json"))
            # Save metrics
            with open(os.path.join(log_dir, "eval_results.json"), "w") as file:
                json.dump(eval_results, file)

            with open(os.path.join(log_dir, "test_results.json"), "w") as file:
                json.dump(test_results, file)
        else:
            # Parallel
            # model = nn.DataParallel(model)
            
            for epoch in range(args.max_epochs):
                torch.cuda.empty_cache()
                start_time = time.time()

                train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, scl=args.scl, gan=args.gan, gan_batch_size=args.gan_batch_size, gan_data_path=args.gan_data_path)
                valid_loss, valid_metrics = test(model, valid_loader, criterion, device)

                scheduler.step()
                if epoch < args.warm_epochs:                
                    current_clf_lr = scheduler.get_last_lr()[-1]
                else:
                    current_clf_lr = scheduler.after_scheduler.get_last_lr()[-1]

                end_time = time.time()
                epoch_secs = end_time - start_time

                valid_acc = float(valid_metrics['Accuracy'])
                valid_f1 = float(valid_metrics['F1-score'])
                valid_mcc = float(valid_metrics['MCC'])
                valid_map = float(valid_metrics['AUPRC'])
                valid_auc = float(valid_metrics['AUC'])
                valid_precision = float(valid_metrics['Precision'])
                valid_recall = float(valid_metrics['Recall'])
                valid_specificity = float(valid_metrics['Specificity'])
                valid_average_precision = float(valid_metrics['average_precision'])
                valid_fpr = valid_metrics['fpr']
                valid_fpr = [str(i) for i in valid_fpr]
                valid_fpr = " ".join(valid_fpr)
                valid_tpr = valid_metrics['tpr']
                valid_tpr = [str(i) for i in valid_tpr]
                valid_tpr = " ".join(valid_tpr)
                valid_precision_array = valid_metrics['precision_array']
                valid_precision_array = [str(i) for i in valid_precision_array]
                valid_precision_array = " ".join(valid_precision_array)
                valid_recall_array = valid_metrics['recall_array']
                valid_recall_array = [str(i) for i in valid_recall_array]
                valid_recall_array = " ".join(valid_recall_array)

                logging.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
                logging.info(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.1f}%')
                logging.info(f'Loss: {valid_loss:.3f} | Acc: {valid_acc*100:.1f}% | Precision: {valid_precision*100:.1f}% | Recall: {valid_recall*100:.1f}% | Specificity: {valid_specificity*100:.1f}% | F1: {valid_f1:.3f} | MCC: {valid_mcc:.3f} | AUC: {valid_auc:.3f} | mAP: {valid_map:.3f} | average_precision: {valid_average_precision:.3f} | lr: {current_clf_lr} | fpr: {valid_fpr} | tpr: {valid_tpr} | precision_array: {valid_precision_array} | recall_array: {valid_recall_array}')

                #writer.add_scalar('Train/Loss', train_loss, epoch+1)
                #writer.add_scalar('Train/Accuracy', train_acc, epoch+1)
                #writer.add_scalar('Valid/Loss', valid_loss, epoch+1)
                #for key, value in valid_metrics.items():
                #    writer.add_scalar('Valid/' + key, value, epoch+1)

                early_stopping(valid_f1, model)
                if early_stopping.early_stop:
                    logging.info(f"Early stopping at Epoch {epoch+1}")
                    break        
            
            # Test using the best model
            model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoint.pt'))) # Load the best model
            test_loss, test_metrics = test(model, test_loader, criterion, device)
            test_acc = float(test_metrics['Accuracy'])
            test_f1 = float(test_metrics['F1-score'])
            test_mcc = float(test_metrics['MCC'])
            test_map = float(test_metrics['AUPRC'])
            test_auc = float(test_metrics['AUC'])
            test_precision = float(test_metrics['Precision'])
            test_recall = float(test_metrics['Recall'])
            test_specificity = float(test_metrics['Specificity'])
            test_average_precision = float(test_metrics['average_precision'])
            test_fpr = test_metrics['fpr']
            test_fpr = [str(i) for i in test_fpr]
            test_fpr = " ".join(test_fpr)
            test_tpr = test_metrics['tpr']
            test_tpr = [str(i) for i in test_tpr]
            test_tpr = " ".join(test_tpr)
            test_precision_array = test_metrics['precision_array']
            test_precision_array = [str(i) for i in test_precision_array]
            test_precision_array = " ".join(test_precision_array)
            test_recall_array = test_metrics['recall_array']
            test_recall_array = [str(i) for i in test_recall_array]
            test_recall_array = " ".join(test_recall_array)
            
            logging.info(f'Loss: {test_loss:.3f} | Acc: {test_acc*100:.1f}% | Precision: {test_precision*100:.1f}% | Recall: {test_recall*100:.1f}% | Specificity: {test_specificity*100:.1f}% | F1: {test_f1:.3f} | MCC: {test_mcc:.3f} | AUC: {test_auc:.3f} | mAP: {test_map:.3f} | average_precision: {test_average_precision:.3f} | fpr: {test_fpr} | tpr: {test_tpr} | precision_array: {test_precision_array} | recall_array: {test_recall_array}')


    # Testing
    if args.mode == 'test':
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'checkpoint.pt')))
        model = nn.DataParallel(model) # Parallel
        #valid_best_loss, valid_best_metrics, valid_truth, valid_pred = test(model, valid_loader, criterion, device, True)
        test_loss, test_metrics, test_truth, test_prob, test_pred = test(model, test_loader, criterion, device, return_array=True)

        test_acc = float(test_metrics['Accuracy'])
        test_f1 = float(test_metrics['F1-score'])
        test_mcc = float(test_metrics['MCC'])
        test_map = float(test_metrics['AUPRC'])
        test_auc = float(test_metrics['AUC'])
        test_precision = float(test_metrics['Precision'])
        test_recall = float(test_metrics['Recall'])
        test_specificity = float(test_metrics['Specificity'])
        test_average_precision = float(test_metrics['average_precision'])
        test_fpr = test_metrics['fpr']
        test_fpr = [str(i) for i in test_fpr]
        test_fpr = " ".join(test_fpr)
        test_tpr = test_metrics['tpr']
        test_tpr = [str(i) for i in test_tpr]
        test_tpr = " ".join(test_tpr)
        test_precision_array = test_metrics['precision_array']
        test_precision_array = [str(i) for i in test_precision_array]
        test_precision_array = " ".join(test_precision_array)
        test_recall_array = test_metrics['recall_array']
        test_recall_array = [str(i) for i in test_recall_array]
        test_recall_array = " ".join(test_recall_array)
        
        #logging.info(f'Best Valid Loss: {valid_best_loss:.3f} | Acc: {valid_best_metrics["Accuracy"]*100:.2f}% |'
        #            f' F1: {valid_best_metrics["F1-score"]:.3f} | mAP: {valid_best_metrics["AUPRC"]:.3f}')
        logging.info(f'Loss: {test_loss:.3f} | Acc: {test_acc*100:.1f}% | Precision: {test_precision*100:.1f}% | Recall: {test_recall*100:.1f}% | Specificity: {test_specificity*100:.1f}% | F1: {test_f1:.3f} | MCC: {test_mcc:.3f} | AUC: {test_auc:.3f} | mAP: {test_map:.3f} | average_precision: {test_average_precision:.3f} | fpr: {test_fpr} | tpr: {test_tpr} | precision_array: {test_precision_array} | recall_array: {test_recall_array}')

        #for key, value in valid_best_metrics.items():
        #    writer.add_scalar('Valid/Best ' + key, value)
        #for key, value in test_final_metrics.items():
        #    writer.add_scalar('Test/Final ' + key, value)

    # Prediction
    if args.mode == 'predict':
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'checkpoint.pt')))       
        model = nn.DataParallel(model) # Parallel
        #valid_best_loss, valid_best_metrics, valid_truth, valid_pred = test(model, valid_loader, criterion, device, True)
        if args.save_attn:
            test_truth, test_prob, test_pred, attn_dict = test(model, test_loader, criterion, device, predict=True, save_attn=args.save_attn)
        else:
            test_truth, test_prob, test_pred = test(model, test_loader, criterion, device, predict=True, save_attn=args.save_attn)

        output_test_file = os.path.join(log_dir, 'test_result.txt')
        with open(output_test_file, 'w') as f:
            f.write('index\tprediction\tprobability\n')
            for i in range(0, len(test_pred)):
                f.write('%d\t%d\t%.3f\n' % (i, test_pred[i], test_prob[i]))

        if args.save_attn:
            print(f"Saving attention in {os.path.join(log_dir, 'attn.npz')}")
            np.savez(os.path.join(log_dir, 'attn.npz'), **attn_dict)

    #valid_cm = confusion_matrix(valid_truth, valid_pred)
    #test_cm = confusion_matrix(test_truth, test_pred)

    #labels = ['Non-effector', 'T1SE', 'T2SE', 'T3SE', 'T4SE', 'T6SE']

    #writer.add_figure('Valid/conf_matrix', viz_conf_matrix(valid_cm, labels))
    #writer.add_figure('Test/conf_matrix', viz_conf_matrix(test_cm, labels))
    #writer.close()


if __name__ == '__main__':

    parser = ArgumentParser(description="Train a TAfinder2 model for type II TA system prediction.")
    
    # Select Model
    parser.add_argument('--model', choices=['esm2model', 'esm2model_simple', 'esm2model_finetune_notshareparams', 'esm2model_XGBoost'], type=str,
                        help="model types available for training. ")
    parser.add_argument('--model_dir', type=str,
                        help="model directory for test.")

    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int,
                        help="bacth size used in training. (default: 32)")
    parser.add_argument('--num_workers', default=4, type=int,
                        help="The number of workers used in dataloader")
    parser.add_argument('--seed', default=42, type=int, 
                        help="random seed used in training. (default: 42)")
    parser.add_argument('--lr', default=1e-4, type=float,
                        help="learning rate of classifier. (default: 1e-4)")
    parser.add_argument('--enc_lr', default=1e-4, type=float,
                        help="learning rate of pre-trained model. (default: 1e-4)")
    parser.add_argument('--warm_epochs', default=1, type=int,
                        help="The number of epochs under warm start. (default: 1)")
    parser.add_argument('--patience', default=5, type=int,
                        help="The number of patience epochs/steps for early stopping. (default: 5)")
    parser.add_argument('--lr_scheduler', default='exp', choices=['step', 'cosine', 'exp'], type=str,
                        help="learning rate scheduler. [step, cosine, exp]")
    parser.add_argument('--lr_decay_steps', default=10, type=int, 
                        help="step of learning rate decay. Used when lr_scheduler == 'step' or 'cos'. (default: 10)")
    parser.add_argument('--lr_decay_rate', default=0.5, type=float,
                        help="ratio of learning rate decay. Used when lr_scheduler == 'exp'. (default: 0.5)")
    parser.add_argument('--lr_decay_min_lr', default=5e-6, type=float,
                        help="minimum value of learning rate. Used when lr_scheduler == 'cos'. (default: 5e-6)")

    # Training Info
    parser.add_argument('--mode', choices=['train', 'test', 'predict'], type=str,
                        help="Mode. ['train', 'test', 'predict']")
    parser.add_argument('--max_epochs', default=30, type=int,
                        help="Maximum number of epochs. (default: 30)")
    parser.add_argument('--num_layers', default=0, type=int,
                        help="The number of layers to be fine-tuned in pre-trained model. (default: 0)")
    parser.add_argument('--repr_layers', default=33, type=int,
                        help="Layers indices from which to extract representations. (default: 33)")
    parser.add_argument('--data_dir', default='./data', type=str,
                        help="path to data. (default: ./data)")
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help="weight decay for regularization. (default: 1e-5)")
    parser.add_argument('--log_dir', default='./logs', type=str,
                        help="path to the logging directory. (default: ./logs)")
    parser.add_argument('--plot_embedding', action = "store_true", required = False, default = False,
                        help="Whether only plot sequence embeddings. (default: False)")
    parser.add_argument('--get_embedding', action = "store_true", required = False, default = False,
                        help="Only extract sequence embeddings. (default: False)")
    parser.add_argument('--embedding_file', default='./embedding.txt', required = False, type=str,
                        help="path to store embeddings. Only used with --get_embedding parameter. (default: ./embedding.txt)")
    parser.add_argument('--scl', action = "store_true", required = False, default = False,
                        help="Whether to add supervised contrastive loss. (default: False)")
    parser.add_argument('--gan', action = "store_true", required = False, default = False,
                        help="Whether to use GAN-generated positive samples. (default: False)")
    parser.add_argument('--gan_batch_size', required = False, default = 5, type=int,
                        help="The number of GAN-generated positive samples in each batch when using GAN. (default: 5)")
    parser.add_argument('--gan_data_path', required = False, type=str,
                        help="The path of GAN-generated positive sample file.")
    parser.add_argument('--stratified', action = "store_true", required = False, default = False,
                        help="Whether to use stratified sampling per batch. (default: False)")
    parser.add_argument('--stratified_TA_non_TA', action = "store_true", required = False, default = False,
                        help="Whether to use stratified sampling, and include the corresponding TA-nonTA negative samples per batch. (default: False)")
    parser.add_argument('--return_attn', action='store_true', required = False, default = False, 
                        help='Wherther model return the sequence attention.')
    parser.add_argument('--save_attn', action='store_true', required = False, default = False, 
                        help='Do not save the sequence attention.')
    parser.add_argument('--keep_seq_len', action='store_true', required = False, default = False, 
                        help='Return the tensors without taking the mean value for the seq_len dimension.')

    # Model Hyperparameters
    parser.add_argument('--hid_dim', default=256, type=int,
                        help="hidden dimension in the model. (default: 256)")
    parser.add_argument('--dropout_rate', default=0.4, type=float,
                        help="dropout rate (default: 0.4)")

    args = parser.parse_args()

    if args.save_attn:
        args.save_attn = False
    else:
        args.save_attn = True

    main(args)

