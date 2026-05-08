import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from einops import rearrange
import esm
from DeepTAfinder.module import MLPLayer, TransformerLayer
from torch.nn.utils.weight_norm import weight_norm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score, precision_recall_curve, roc_curve


class ESM2Model(nn.Module):
    def __init__(self, emb_dim, repr_layer,
                    num_layers, hid_dim=256,
                    dropout_rate=0.4, num_classes=1,
                    return_embedding=False,
                    return_attn=False,
                    keep_seq_len=False):

        super().__init__()
        self.pretrained_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.padding_idx = alphabet.padding_idx
        self.repr_layer = repr_layer
        self.num_layers = num_layers # Number of last layers to fine-tune        
        self.conv_1 = nn.Conv1d(emb_dim, 512, 1, 1, bias=False)
        self.conv_2 = nn.Conv1d(emb_dim, 512, 1, 1, bias=False)
        self.transformer_layer_1 = TransformerLayer(hid_dim=512, heads=8, dropout_rate=0.1)
        self.transformer_layer_2 = TransformerLayer(hid_dim=512, heads=8, dropout_rate=0.1)
        self.clf = MLPLayer(in_dim=512*2, hid_dim=hid_dim, num_classes=num_classes, dropout_rate=dropout_rate)
        self.return_attn = return_attn

        # Fine-tune the last 'num_layers' layers.
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        for name, param in self.named_parameters():
            if name.startswith('pretrained_model.layers'):
                if int(name.split('.')[2]) >= (self.repr_layer - self.num_layers):
                    param.requires_grad = True
        
        self.return_embedding = return_embedding
        self.keep_seq_len = keep_seq_len

    def forward(self, strs_1, strs_2, toks_1, toks_2, GAN_feature=torch.Tensor(), GAN_feature_mask_1=torch.Tensor(), GAN_feature_mask_2=torch.Tensor()):
        toks_1 = toks_1[:, :1024]
        toks_2 = toks_2[:, :1024]
        padding_mask_1 = (toks_1 != self.padding_idx)[:, 1:-1]
        padding_mask_2 = (toks_2 != self.padding_idx)[:, 1:-1]
        batch = toks_1.shape[0] # toks_1.shape is equal to toks_2.shape        
        out_1 = self.pretrained_model(toks_1, repr_layers=[self.repr_layer], return_contacts=False)  # Obtain final-layer embeddings of pre-trained model
        out_1 = out_1["representations"][self.repr_layer][:, 1:-1, :] # (bs, seq_len, emb_dim)        
        if GAN_feature.numel() != 0:
            GAN_feature_1 = GAN_feature[:, :out_1.shape[1], :1280] # 前1280个emb_dim
            out_1 = torch.cat([out_1, GAN_feature_1], dim=0) # concat            
            padding_mask_1 = torch.cat([padding_mask_1, GAN_feature_mask_1], dim=0) # GAN样本的mask与真实样本的mask放一起
            #order = torch.randperm(out_1.shape[0]).to(device) # shuffle
            #out_1 = torch.index_select(out_1, 0, order)
            out_1 = out_1.float()
            batch = out_1.shape[0]
        out_1 = out_1 * padding_mask_1.unsqueeze(-1).type_as(out_1)
        out_1 = rearrange(out_1, 'b n d -> b d n')
        out_1 = self.conv_1(out_1)  # dimension reduction
        out_1 = rearrange(out_1, 'b d n -> b n d')
        out_1, attn_1 = self.transformer_layer_1(out_1, mask=padding_mask_1.unsqueeze(1).unsqueeze(2)) # (bs, seq_len, emb_dim)
        valid_lengths_1 = padding_mask_1.sum(dim=1)  # (batch_size,)
        emb_1 = out_1.sum(dim=1) / valid_lengths_1.unsqueeze(-1) # Generate per-sequence representations via averaging

        out_2 = self.pretrained_model(toks_2, repr_layers=[self.repr_layer], return_contacts=False)  # Obtain final-layer embeddings of pre-trained model
        out_2 = out_2["representations"][self.repr_layer][:, 1:-1, :] # (bs, seq_len, emb_dim)        
        if GAN_feature.numel() != 0:
            GAN_feature_2 = GAN_feature[:, :out_2.shape[1], 1280:] # 后1280个emb_dim
            out_2 = torch.cat([out_2, GAN_feature_2], dim=0) # concat            
            padding_mask_2 = torch.cat([padding_mask_2, GAN_feature_mask_2], dim=0) # GAN样本的mask与真实样本的mask放一起
            #order = torch.randperm(out_2.shape[0]).to(device) # shuffle
            #out_2 = torch.index_select(out_2, 0, order)
            out_2 = out_2.float()
            batch = out_2.shape[0]
        out_2 = out_2 * padding_mask_2.unsqueeze(-1).type_as(out_2)
        out_2 = rearrange(out_2, 'b n d -> b d n')
        out_2 = self.conv_2(out_2)  # dimension reduction
        out_2 = rearrange(out_2, 'b d n -> b n d')
        out_2, attn_2 = self.transformer_layer_2(out_2, mask=padding_mask_2.unsqueeze(1).unsqueeze(2)) # (bs, seq_len, emb_dim)
        valid_lengths_2 = padding_mask_2.sum(dim=1)  # (batch_size,)
        emb_2 = out_2.sum(dim=1) / valid_lengths_2.unsqueeze(-1) # Generate per-sequence representations via averaging
        
        emb = torch.cat([emb_1, emb_2], dim=1) # Concatenation (bs, emb_dim * 2)
        #emb = torch.multiply(emb_1, emb_2) # Element-wise multiplication (bs, emb_dim)
        
        if self.keep_seq_len:
            return torch.cat([out_1, out_2], dim=2)
        elif self.return_embedding:
            return emb
        else:
            logits = self.clf(emb)
            if self.return_attn:
                attn = [attn_1, attn_2]
                return logits, emb, attn
            else:
                return logits, emb


class ESM2FeatureExtractor(nn.Module):
    def __init__(self, emb_dim, repr_layer):

        super().__init__()
        self.pretrained_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.padding_idx = alphabet.padding_idx
        self.repr_layer = repr_layer

        # Freeze all layers
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, toks_1, toks_2):
        toks_1 = toks_1[:, :1024]
        toks_2 = toks_2[:, :1024]
        padding_mask_1 = (toks_1 != self.padding_idx)[:, 1:-1]
        padding_mask_2 = (toks_2 != self.padding_idx)[:, 1:-1]
       
        out_1 = self.pretrained_model(toks_1, repr_layers=[self.repr_layer], return_contacts=False)  # Obtain final-layer embeddings of pre-trained model
        out_1 = out_1["representations"][self.repr_layer][:, 1:-1, :] # (bs, seq_len, emb_dim)        
        out_1 = out_1 * padding_mask_1.unsqueeze(-1).type_as(out_1)        
        valid_lengths_1 = padding_mask_1.sum(dim=1)  # (batch_size,)
        emb_1 = out_1.sum(dim=1) / valid_lengths_1.unsqueeze(-1) # Generate per-sequence representations via averaging

        out_2 = self.pretrained_model(toks_2, repr_layers=[self.repr_layer], return_contacts=False)  # Obtain final-layer embeddings of pre-trained model
        out_2 = out_2["representations"][self.repr_layer][:, 1:-1, :] # (bs, seq_len, emb_dim)
        out_2 = out_2 * padding_mask_2.unsqueeze(-1).type_as(out_2)
        valid_lengths_2 = padding_mask_2.sum(dim=1)  # (batch_size,)
        emb_2 = out_2.sum(dim=1) / valid_lengths_2.unsqueeze(-1) # Generate per-sequence representations via averaging

        emb = torch.cat([emb_1, emb_2], dim=1) # Concatenation (bs, emb_dim * 2)
        
        return emb


class ESM2XGBoostClassifier:
    def __init__(self, emb_dim, device, repr_layer=33, xgb_params=None):
        self.feature_extractor = ESM2FeatureExtractor(emb_dim, repr_layer)
        self.xgb_model = None
        self.device = device        

        # Set XGBoost default parameters
        if xgb_params is None:
            self.xgb_params = {
                'max_depth': 6,
                'learning_rate': 0.01,
                'n_estimators': 1000,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'early_stopping_rounds': 10,
                'device': 'gpu'
            }
        else:
            self.xgb_params = xgb_params
    
    def extract_features(self, dataloader):
        all_features = []
        all_labels = []
        
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        with torch.no_grad():            
            for labels, strs, toks in dataloader:
                toks_1, toks_2 = zip(*toks)
                toks_1 = torch.stack(toks_1).to(self.device)
                toks_2 = torch.stack(toks_2).to(self.device)
                features = self.feature_extractor(toks_1, toks_2)
                all_features.append(features.cpu().numpy())
                all_labels.append(np.array(labels))
        
        return np.vstack(all_features), np.concatenate(all_labels)
    
    def train(self, train_loader, valid_loader):
        # Extract training features
        X_train, y_train = self.extract_features(train_loader)
        
        # Extract validation features
        X_val, y_val = self.extract_features(valid_loader)
        eval_set = [(X_val, y_val)]
        
        # Train XGBoost model
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,            
            verbose=True
        )
        
        return self.xgb_model
    
    def predict(self, dataloader):
        if self.xgb_model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Extract features
        X, y = self.extract_features(dataloader)
        
        # Prediction
        y_pred = self.xgb_model.predict(X)
        y_prob = self.xgb_model.predict_proba(X)[:, 1]  # Get positive sample probabilities
        
        return y, y_pred, y_prob
    
    def evaluate(self, dataloader):
        y_true, y_pred, y_prob = self.predict(dataloader)
        
        accuracy = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        fpr, tpr, thresholds_1 = roc_curve(y_true, y_prob, pos_label=1)
        precision_array, recall_array, thresholds_2 = precision_recall_curve(y_true, y_prob, pos_label=1)
        auc = roc_auc_score(y_true, y_prob)
        average_precision = average_precision_score(y_true, y_prob)
        
        return {
            "Accuracy": accuracy, 
            "Precision": precision, 
            "Recall": recall, 
            "Specificity": specificity, 
            "F1-score": f1,
            "MCC": mcc, 
            "AUC": auc,
            "AUPRC": average_precision,
            'precision_array': precision_array.tolist(),
            'recall_array': recall_array.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'average_precision': average_precision
        }
    
    def save_model(self, path):
        if self.xgb_model is None:
            raise ValueError("No model to save.")
        self.xgb_model.save_model(path)
    
    def load_model(self, path):
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(path)


class ESM2ModelFinetuneNotShareParams(nn.Module):
    def __init__(self, emb_dim, repr_layer,
                    num_layers, hid_dim=256,
                    dropout_rate=0.4, num_classes=1,
                    return_embedding=False,
                    return_attn=False,
                    keep_seq_len=False):

        super().__init__()
        self.pretrained_model_1, alphabet_1 = esm.pretrained.esm2_t33_650M_UR50D()
        self.pretrained_model_2, alphabet_2 = esm.pretrained.esm2_t33_650M_UR50D()
        self.padding_idx = alphabet_1.padding_idx
        self.repr_layer = repr_layer
        self.num_layers = num_layers # Number of last layers to fine-tune
        self.clf = MLPLayer(in_dim=2560, hid_dim=hid_dim, num_classes=num_classes, dropout_rate=dropout_rate)
        self.return_attn = return_attn

        # Fine-tune the last 'num_layers' layers.
        for param in self.pretrained_model_1.parameters():
            param.requires_grad = False

        for name, param in self.named_parameters():
            if name.startswith('pretrained_model_1.layers'):
                if int(name.split('.')[2]) >= (self.repr_layer - self.num_layers):
                    param.requires_grad = True

        for param in self.pretrained_model_2.parameters():
            param.requires_grad = False

        for name, param in self.named_parameters():
            if name.startswith('pretrained_model_2.layers'):
                if int(name.split('.')[2]) >= (self.repr_layer - self.num_layers):
                    param.requires_grad = True
        
        self.return_embedding = return_embedding
        self.keep_seq_len = keep_seq_len

    def forward(self, strs_1, strs_2, toks_1, toks_2, GAN_feature=torch.Tensor()):
        toks_1 = toks_1[:, :1024]
        toks_2 = toks_2[:, :1024]
        padding_mask_1 = (toks_1 != self.padding_idx)[:, 1:-1]
        padding_mask_2 = (toks_2 != self.padding_idx)[:, 1:-1]
        batch = toks_1.shape[0] # toks_1.shape is equal to toks_2.shape

        out_1 = self.pretrained_model_1(toks_1, repr_layers=[self.repr_layer], return_contacts=False)  # Obtain final-layer embeddings of pre-trained model
        out_1 = out_1["representations"][self.repr_layer][:, 1:-1, :] # (bs, seq_len, emb_dim)
        out_1 = out_1 * padding_mask_1.unsqueeze(-1).type_as(out_1)
        emb_1 = torch.cat([out_1[i, :len(strs_1[i]) + 1].mean(0).unsqueeze(0) for i in range(batch)], dim=0) # Generate per-sequence representations via averaging

        out_2 = self.pretrained_model_2(toks_2, repr_layers=[self.repr_layer], return_contacts=False)  # Obtain final-layer embeddings of pre-trained model
        out_2 = out_2["representations"][self.repr_layer][:, 1:-1, :] # (bs, seq_len, emb_dim)
        out_2 = out_2 * padding_mask_2.unsqueeze(-1).type_as(out_2)
        emb_2 = torch.cat([out_2[i, :len(strs_2[i]) + 1].mean(0).unsqueeze(0) for i in range(batch)], dim=0) # Generate per-sequence representations via averaging
    
        emb = torch.cat([emb_1, emb_2], dim=1) # Concatenation (bs, emb_dim * 2)
        #emb = torch.multiply(emb_1, emb_2) # Element-wise multiplication (bs, emb_dim)
        
        if self.keep_seq_len:
            return torch.cat([out_1, out_2], dim=2)
        elif self.return_embedding:
            return emb
        else:
            logits = self.clf(emb)
            return logits, emb
            
            
class ESM2ModelSimple(nn.Module):
    def __init__(self, emb_dim, repr_layer,
                    num_layers, hid_dim=256,
                    dropout_rate=0.4, num_classes=2,
                    return_embedding=False, keep_seq_len=False):

        super().__init__()
        self.pretrained_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()        
        self.repr_layer = repr_layer
        self.padding_idx = alphabet.padding_idx
        self.num_layers = num_layers # Number of last layers to fine-tune     
        self.clf = MLPLayer(in_dim=512, hid_dim=hid_dim, num_classes=num_classes, dropout_rate=dropout_rate)
        
        # Fine-tune the last 'num_layers' layers.
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        for name, param in self.named_parameters():
            if name.startswith('pretrained_model.layers'):
                if int(name.split('.')[2]) >= (self.repr_layer - self.num_layers):
                    param.requires_grad = True
        
        self.return_embedding = return_embedding
        self.keep_seq_len = keep_seq_len

    def forward(self, strs_1, strs_2, toks_1, toks_2):
        toks_1 = toks_1[:, :1024]
        toks_2 = toks_2[:, :1024]
        padding_mask_1 = (toks_1 != self.padding_idx)[:, 1:-1]
        padding_mask_2 = (toks_2 != self.padding_idx)[:, 1:-1]

        out_1 = self.pretrained_model(toks_1, repr_layers=[self.repr_layer], return_contacts=False)  # Obtain final-layer embeddings of pre-trained model (bs, seq_len, emb_dim)
        out_1 = out_1["representations"][self.repr_layer][:, 1:-1, :]
        out_1 = out_1 * padding_mask_1.unsqueeze(-1).type_as(out_1)
        valid_lengths_1 = padding_mask_1.sum(dim=1)  # (batch_size,)
        emb_1 = out_1.sum(dim=1) / valid_lengths_1.unsqueeze(-1) # Generate per-sequence representations via averaging

        out_2 = self.pretrained_model(toks_2, repr_layers=[self.repr_layer], return_contacts=False)  # Obtain final-layer embeddings of pre-trained model (bs, seq_len, emb_dim)
        out_2 = out_2["representations"][self.repr_layer][:, 1:-1, :]
        out_2 = out_2 * padding_mask_2.unsqueeze(-1).type_as(out_2)
        valid_lengths_2 = padding_mask_2.sum(dim=1)  # (batch_size,)
        emb_2 = out_2.sum(dim=1) / valid_lengths_2.unsqueeze(-1) # Generate per-sequence representations via averaging

        emb = torch.cat([emb_1, emb_2], dim=1) # Concatenation (bs, emb_dim * 2)
        #emb = torch.multiply(emb_1, emb_2) # Element-wise multiplication (bs, emb_dim)

        if self.keep_seq_len:
            return torch.cat([out_1, out_2], dim=2)
        elif self.return_embedding:
            return emb
        else:
            logits = self.clf(emb)
            return logits, emb