import os
import random

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, recall_score, precision_score, roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, average_precision_score

from DeepTAfinder.utils import metrics
from DeepTAfinder.losses import contrastive_loss

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
class EarlyStopping:

    def __init__(self, patience=10, checkpoint_dir='logs'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, score, model, goal="maximize"):

        if goal == "minimize":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)

        #elif score < self.best_score:
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        
        # Only start counting when score != 0
        if score == 0:
            self.counter = 0
            self.best_score = None

    def save_checkpoint(self, score, model):
        torch.save(model.state_dict(), os.path.join(
            self.checkpoint_dir, 'checkpoint.pt'))
        self.best_score = score


def generate_padding_mask(lengths, max_length=1022):
        """
        根据蛋白序列长度列表生成 padding_mask 矩阵
        mask 位置为 0，有效位置为 1
        
        参数：
            lengths (torch.Tensor()) : 各蛋白序列的实际长度
            max_length (torch.Tensor()) : 填充后的总长度（默认 1022）
        
        返回：
            np.ndarray : 形状为 (num_sequences, max_length) 的 mask 矩阵
        """
        # 验证数据范围
        assert torch.all(lengths <= max_length), "存在长度超过 max_length 的序列"
        
        # 生成列索引 [0, 1, 2, ..., max_length-1]
        column_indices = torch.arange(max_length)

        # 比较长度
        mask = column_indices < lengths.unsqueeze(-1)
        
        return torch.Tensor(mask)


def train(model, iterator, criterion, optimizer, device, gan_batch_size=0, gan_data_path='', scl=False, gan=False):

    avg_loss = 0
    avg_acc = 0
    data_size = 0
    
    model.train()
    scaler = GradScaler()

    if gan:
        GAN_data = torch.load(gan_data_path)
        GAN_feature = GAN_data['generate_sample']
        gen_anti_lens = GAN_data['gen_anti_lens']
        gen_tox_lens = GAN_data['gen_tox_lens']
        #np.random.shuffle(GAN_data) 
        #GAN_data = torch.from_numpy(GAN_data)
        n = 0

    for labels, strs, toks in iterator:
        optimizer.zero_grad()

        strs_1, strs_2 = zip(*strs)
        toks_1, toks_2 = zip(*toks)
        toks_1 = torch.stack(toks_1).to(device)
        toks_2 = torch.stack(toks_2).to(device)    
        
        with autocast(dtype=torch.float16):
            if gan:
                GAN_feature_input = GAN_feature[n:(n+gan_batch_size), :, :]
                gen_anti_lens_input = torch.tensor(gen_anti_lens[n:(n+gan_batch_size)])
                gen_tox_lens_input = torch.tensor(gen_tox_lens[n:(n+gan_batch_size)])
                GAN_feature_mask_1 = generate_padding_mask(gen_anti_lens_input)
                GAN_feature_mask_2 = generate_padding_mask(gen_tox_lens_input)
                #random_indices = np.random.randint(0, GAN_data.shape[0], size=gan_batch_size)
                #GAN_data_input = GAN_data[random_indices, :, :]
                GAN_feature_input = GAN_feature_input.to(device)
                GAN_feature_mask_1 = GAN_feature_mask_1.to(device)
                GAN_feature_mask_2 = GAN_feature_mask_2.to(device)

                logits, embedding = model(strs_1, strs_2, toks_1, toks_2, GAN_feature_input, GAN_feature_mask_1, GAN_feature_mask_2)
                n = n + gan_batch_size
            else:
                logits, embedding = model(strs_1, strs_2, toks_1, toks_2)
            # 获取正样本的logits，作为BCELoss的输入
            #logits_positive = logits[:, 1]
            logits_positive = logits.flatten()
            #logits_positive = logits.squeeze()
            if gan:
                """
                if len(logits_positive) < 32:
                    y = torch.tensor(labels + (1,) * (gan_batch_size // 2), device=device).long() # 用两张显卡的情况下，最后一个batch的gan_batch_size要除以2
                else:
                    y = torch.tensor(labels + (1,) * gan_batch_size, device=device).long()
                """
                y = torch.tensor(labels + (1,) * gan_batch_size, device=device).long()
            else:
                y = torch.tensor(labels, device=device).long()

            #_, pred = torch.max(logits.data, 1)
            threshold = 0.5
            prob = torch.sigmoid(logits_positive.float())
            pred = torch.tensor([1 if i > threshold else 0 for i in prob])
            weights = torch.ones_like(y.float()) # loss 的初始权重都为 1
            if gan:
                weights[-gan_batch_size:] *= 1  # 如果用GAN，设置GAN样本的权重

            if scl == True:
                lam = 0.9
                tem = 0.3
                # cross_loss = criterion(logits, y, weights)
                cross_loss = criterion(logits_positive, y, weights) # 对于 BCE 损失函数
                contra_loss = contrastive_loss(embedding.cpu().detach().numpy(), y, tem)
                loss = (lam * contra_loss) + (1 - lam) * (cross_loss)
            else:   
                # loss = criterion(logits, y, weights)
                loss = criterion(logits_positive, y, weights) # 对于 BCE 损失函数

        # 反向传播与优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = accuracy_score(y.cpu().numpy(), pred.cpu().numpy())
        avg_loss += loss.cpu().item() * len(labels)
        avg_acc += acc * len(labels)
        data_size += len(labels)

    avg_acc = avg_acc / data_size
    avg_loss = avg_loss / data_size

    return avg_loss, avg_acc


def test(model, iterator, criterion, device, return_array=False, predict=False, save_attn=False):

    avg_loss = 0
    data_size = 0

    truth = []
    probs = []
    preds = []

    model.eval()

    if save_attn:
        model.return_attn = True
        attn_dict = {}
        
    with torch.no_grad():
        for labels, strs, toks in iterator:
            strs_1, strs_2 = zip(*strs)
            toks_1, toks_2 = zip(*toks)
            toks_1 = torch.stack(toks_1).to(device)
            toks_2 = torch.stack(toks_2).to(device)
            if save_attn:
                with autocast():
                    logits, embedding, attn_list = model(strs_1, strs_2, toks_1, toks_2)
                attn_1 = attn_list[0].cpu().numpy() # (seq_len, batch_size, embed_dim)
                attn_2 = attn_list[1].cpu().numpy() # (seq_len, batch_size, embed_dim)
                #print(attn_1.shape)
                #print(attn_2.shape)
                for i, str in enumerate(strs_1):
                    seq = str[:1024]
                    label = f"{labels[i]}_AT"
                    avg_attn = attn_1[i, :, :len(seq), :len(seq)].sum(0).mean(0)
                    #attn_dict[str[:10]] = avg_attn
                    attn_dict[label] = avg_attn
                for i, str in enumerate(strs_2):
                    seq = str[:1024]
                    label = f"{labels[i]}_T"
                    avg_attn = attn_2[i, :, :len(seq), :len(seq)].sum(0).mean(0)
                    #attn_dict[str[:10]] = avg_attn
                    attn_dict[label] = avg_attn
            else:
                with autocast():
                    logits, embedding = model(strs_1, strs_2, toks_1, toks_2)
            # 获取正样本的logits，作为BCELoss的输入
            #logits_positive = logits[:, 1]
            logits_positive = logits.flatten()
            #logits_positive = logits.squeeze()
            y = torch.tensor(labels, device=device).long()
            
            #prob = torch.softmax(logits, dim=1)
            #_, pred = torch.max(prob, 1)
            
            threshold = 0.5            
            prob = torch.sigmoid(logits_positive)
            pred = torch.tensor([1 if i > threshold else 0 for i in prob])

            truth.append(y)
            probs.append(prob)
            preds.append(pred)

            # weights = torch.ones_like(y.float()) # loss 的初始权重都为 1

            # loss = criterion(logits, y, weights)
            # loss = criterion(logits_positive, y, weights) # 对于 BCE 损失函数

            # avg_loss += loss.cpu().item() * len(labels)
            # data_size += len(labels)

    # avg_loss = avg_loss / data_size
    avg_loss = 0

    truth = torch.cat(truth).cpu().numpy()
    probs = torch.cat(probs).cpu().numpy()
    #probs = probs[:, 1]
    preds = torch.cat(preds).cpu().numpy()

    if predict:
        if save_attn:
            return truth, probs, preds, attn_dict
        else:
            return truth, probs, preds

    if return_array:
        metrics_dict = metrics(truth, preds, probs)
        return avg_loss, metrics_dict, truth, probs, preds
    else:
        metrics_dict = metrics(truth, preds, probs)
        return avg_loss, metrics_dict


def train_single(model, iterator, criterion, optimizer, device, gan_batch_size=0, gan_data_path='', scl=False, gan=False):

    avg_loss = 0
    avg_acc = 0
    data_size = 0
    
    model.train()

    if gan:
        GAN_data = np.load(gan_data_path)
        GAN_data = GAN_data['output']
        #np.random.shuffle(GAN_data)        
        GAN_data = torch.from_numpy(GAN_data)
        n = 0

    for labels, strs, toks in iterator:
        toks = toks.to(device)       

        if gan:
            GAN_data_input = GAN_data[n:(n+gan_batch_size), :, :]
            #random_indices = np.random.randint(0, GAN_data.shape[0], size=gan_batch_size)
            #GAN_data_input = GAN_data[random_indices, :, :]
            GAN_data_input = GAN_data_input.to(device)
            logits, embedding = model(strs, toks, GAN_data_input)
            n = n + gan_batch_size
        else:
            logits, embedding = model(strs, toks)
        # 获取正样本的logits，作为BCELoss的输入
        #logits_positive = logits[:, 1]
        logits_positive = logits.flatten()
        #logits_positive = logits.squeeze()
        if gan:
            y = torch.tensor(labels + [1,] * gan_batch_size, device=device).long()
        else:
            y = torch.tensor(labels, device=device).long()

        #_, pred = torch.max(logits.data, 1)
        threshold = 0.5
        prob = torch.sigmoid(logits_positive)
        pred = torch.tensor([1 if i > threshold else 0 for i in prob])
        weights = torch.ones_like(y.float()) # loss 的初始权重都为 1
        #if gan:
            #weights[-gan_batch_size:] *= 1  # 如果用GAN，设置GAN样本的权重

        if scl == True:
            lam = 0.9
            tem = 0.3
            # cross_loss = criterion(logits, y, weights)
            cross_loss = criterion(logits_positive, y, weights) # 对于 BCE 损失函数
            contra_loss = contrastive_loss(embedding.cpu().detach().numpy(), y, tem)
            loss = (lam * contra_loss) + (1 - lam) * (cross_loss)
        else:   
            # loss = criterion(logits, y, weights)
            loss = criterion(logits_positive, y, weights) # 对于 BCE 损失函数

        acc = accuracy_score(y.cpu().numpy(), pred.cpu().numpy())

        avg_loss += loss.cpu().item() * len(labels)
        avg_acc += acc * len(labels)
        data_size += len(labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_acc = avg_acc / data_size
    avg_loss = avg_loss / data_size

    return avg_loss, avg_acc


def test_single(model, iterator, criterion, device, return_array=False, predict=False, save_attn=False):

    avg_loss = 0
    data_size = 0

    truth = []
    probs = []
    preds = []

    model.eval()

    with torch.no_grad():
        for labels, strs, toks in iterator:
            toks = toks.to(device)
            logits, embedding = model(strs, toks)
            # 获取正样本的logits，作为BCELoss的输入
            #logits_positive = logits[:, 1]
            logits_positive = logits.flatten()
            #logits_positive = logits.squeeze()
            y = torch.tensor(labels, device=device).long()
            
            #prob = torch.softmax(logits, dim=1)
            #_, pred = torch.max(prob, 1)
            
            threshold = 0.5            
            prob = torch.sigmoid(logits_positive)
            pred = torch.tensor([1 if i > threshold else 0 for i in prob])

            truth.append(y)
            probs.append(prob)
            preds.append(pred)

            weights = torch.ones_like(y.float()) # loss 的初始权重都为 1

            # loss = criterion(logits, y, weights)
            loss = criterion(logits_positive, y, weights) # 对于 BCE 损失函数

            avg_loss += loss.cpu().item() * len(labels)
            data_size += len(labels)

    avg_loss = avg_loss / data_size

    truth = torch.cat(truth).cpu().numpy()
    probs = torch.cat(probs).cpu().numpy()
    #probs = probs[:, 1]
    preds = torch.cat(preds).cpu().numpy()

    if predict:
        return truth, probs, preds

    if return_array:
        metrics_dict = metrics(truth, preds, probs)
        return avg_loss, metrics_dict, truth, probs, preds
    else:
        metrics_dict = metrics(truth, preds, probs)
        return avg_loss, metrics_dict