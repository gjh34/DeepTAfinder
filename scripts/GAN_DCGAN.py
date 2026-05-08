import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
import pandas as pd
import numpy as np
import random
import os
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import argparse
from tqdm import tqdm


# 设置随机种子函数
def set_seed(seed=42):
    # Python随机模块
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时使用
    # CuDNN配置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# 自定义数据集类 (处理padding mask)
class TADataset(Dataset):
    def __init__(self, real_samples, toxin_lengths, antitoxin_lengths):
        """
        real_samples: 真实样本 tensor [num_samples, 1022, 2560]
        toxin_lengths: 每个样本毒素实际长度列表 [num_samples]
        antitoxin_lengths: 每个样本抗毒素实际长度列表 [num_samples]
        """
        self.real_samples = real_samples
        self.toxin_lengths = toxin_lengths
        self.antitoxin_lengths = antitoxin_lengths

    def __len__(self):
        return len(self.real_samples)
    
    def __getitem__(self, idx):
        x = self.real_samples[idx]
        toxin_len = self.toxin_lengths[idx]
        antitoxin_len = self.antitoxin_lengths[idx]

        # 创建padding mask (1表示有效位置，0表示padding)
        toxin_mask = torch.zeros(1022)
        toxin_mask[:toxin_len] = 1
        
        antitoxin_mask = torch.zeros(1022)
        antitoxin_mask[:antitoxin_len] = 1

        return x, toxin_mask, antitoxin_mask


# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim=100, cond_dim=2, target_len=1022, feature_dim=2560):
        super().__init__()
        self.init_size = 64
        self.init_channels = 512

        # 条件嵌入层
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # 联合输入投影
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 128, self.init_size * self.init_channels),
            nn.BatchNorm1d(self.init_size * self.init_channels),
            nn.ReLU()
        )
        
        # 转置卷积层
        self.conv_blocks = nn.Sequential(
            # [batch, 512, 64]
            nn.ConvTranspose1d(self.init_channels, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),  # [batch, 256, 128]
            
            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),  # [batch, 128, 256]
            
            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),  # [batch, 64, 512]
            
            nn.ConvTranspose1d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),  # [batch, 32, 1024]
            
            # 调整到目标长度
            nn.ConvTranspose1d(32, feature_dim, 3, 1, 1, bias=False),  # [batch, 2560, 1024]
            nn.AdaptiveAvgPool1d(target_len)  # [batch, 2560, 1022]
        )

    def forward(self, z, cond):
        # 条件嵌入
        cond_emb = self.cond_emb(cond)

        # 拼接噪声和条件，通过全连接层
        x = torch.cat([z, cond_emb], dim=1)
        x = self.fc(x)
        x = x.view(-1, self.init_channels, self.init_size)
        
        # 通过转置卷积层
        x = self.conv_blocks(x)
        
        # 调整维度为 [batch, seq_len, features]
        return x.transpose(1, 2)
    

# 判别器
class Discriminator(nn.Module):
    def __init__(self, feature_dim=2560, cond_dim=2):
        super().__init__()

        self.cond_emb = nn.Sequential(
            spectral_norm(nn.Linear(cond_dim, 256)),
            nn.LeakyReLU(0.2)
        )

        self.main = nn.Sequential(
            spectral_norm(nn.Conv1d(feature_dim + 16, 512, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv1d(512, 256, 4, 2, 1, bias=False)),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv1d(256, 128, 4, 2, 1, bias=False)),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv1d(128, 64, 4, 2, 1, bias=False)),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(64, 1)
        )

        # 条件投影层
        self.cond_proj = spectral_norm(nn.Conv1d(256, 16, 1))
        
    def forward(self, x, cond):
        # 处理条件
        cond_emb = self.cond_emb(cond)  # [B, 256]
        cond_emb = cond_emb.unsqueeze(2).expand(-1, -1, x.shape[1])  # [B, 256, L]
        cond_emb = self.cond_proj(cond_emb)  # [B, 16, L]

        # 拼接特征和条件
        x = x.transpose(1, 2)  # [B, F, L]
        x = torch.cat([x, cond_emb], dim=1)

        return self.main(x)
    

# MMD计算函数
def compute_mmd(real_samples, fake_samples, real_toxin_lens, real_anti_lens, fake_toxin_lens, fake_anti_lens, gamma=1.0):
    # 提取真实样本的有效部分均值
    real_antitoxin = []
    real_toxin = []
    for i in range(real_samples.shape[0]):
        anti_len = int(real_anti_lens[i])
        tox_len = int(real_toxin_lens[i])
        real_antitoxin.append(real_samples[i, :anti_len, :1280].mean(dim=0))
        real_toxin.append(real_samples[i, :tox_len, 1280:].mean(dim=0))
    real_features = torch.stack([torch.cat([a, t]) for a, t in zip(real_antitoxin, real_toxin)])

    # 提取生成样本的有效部分均值
    fake_antitoxin = []
    fake_toxin = []
    for i in range(fake_samples.shape[0]):
        anti_len = int(fake_anti_lens[i])
        tox_len = int(fake_toxin_lens[i])
        fake_antitoxin.append(fake_samples[i, :anti_len, :1280].mean(dim=0))
        fake_toxin.append(fake_samples[i, :tox_len, 1280:].mean(dim=0))
    fake_features = torch.stack([torch.cat([a, t]) for a, t in zip(fake_antitoxin, fake_toxin)])

    # 计算MMD
    X = real_features.cpu().numpy()
    Y = fake_features.cpu().numpy()
    
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()


# 计算C2ST分类准确率 (留一交叉验证)
def compute_c2st(real_samples, fake_samples, real_toxin_lens, real_anti_lens, fake_toxin_lens, fake_anti_lens, k=5):
    # 提取真实样本的有效部分均值
    real_antitoxin = []
    real_toxin = []
    for i in range(real_samples.shape[0]):
        anti_len = int(real_anti_lens[i])
        tox_len = int(real_toxin_lens[i])
        real_antitoxin.append(real_samples[i, :anti_len, :1280].mean(dim=0))
        real_toxin.append(real_samples[i, :tox_len, 1280:].mean(dim=0))
    real_features = torch.stack([torch.cat([a, t]) for a, t in zip(real_antitoxin, real_toxin)])

    # 提取生成样本的有效部分均值
    fake_antitoxin = []
    fake_toxin = []
    for i in range(fake_samples.shape[0]):
        anti_len = int(fake_anti_lens[i])
        tox_len = int(fake_toxin_lens[i])
        fake_antitoxin.append(fake_samples[i, :anti_len, :1280].mean(dim=0))
        fake_toxin.append(fake_samples[i, :tox_len, 1280:].mean(dim=0))
    fake_features = torch.stack([torch.cat([a, t]) for a, t in zip(fake_antitoxin, fake_toxin)])

    # 合并特征并创建标签
    features = np.concatenate([real_features.cpu().numpy(), fake_features.cpu().numpy()])
    labels = np.concatenate([np.ones(len(real_features)), 
                      np.zeros(len(fake_features))])
    
    # 特征标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # 留一交叉验证
    loo = LeaveOneOut()
    pred_list = []
    real_list = []

    for train_idx, test_idx in loo.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        # 训练分类器
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        # 预测
        pred_y = knn.predict(X_test)
        pred_list.append(pred_y)
        real_list.append(y_test)
    
    return accuracy_score(real_list, pred_list)


# 计算生成样本间的多样性
def compute_diversity(fake_samples, fake_toxin_lens, fake_anti_lens):
    # 提取生成样本的有效部分均值
    fake_antitoxin = []
    fake_toxin = []
    for i in range(fake_samples.shape[0]):
        anti_len = int(fake_anti_lens[i])
        tox_len = int(fake_toxin_lens[i])
        fake_antitoxin.append(fake_samples[i, :anti_len, :1280].mean(dim=0))
        fake_toxin.append(fake_samples[i, :tox_len, 1280:].mean(dim=0))
    fake_features = torch.stack([torch.cat([a, t]) for a, t in zip(fake_antitoxin, fake_toxin)])
    
    # 计算余弦相似度
    features_norm = F.normalize(fake_features, p=2, dim=1)  # [B, 2560]
    similarity = torch.mm(features_norm, features_norm.T)  # [B, B]

    # 排除对角线
    batch_size = features_norm.size(0)
    mask = torch.eye(batch_size, dtype=torch.bool, device=features_norm.device)
    similarity.masked_fill_(mask, 0)

    # 计算多样性
    diversity = 1 - similarity.sum() / (batch_size * (batch_size - 1))

    return diversity.item()


# 梯度惩罚计算函数
def compute_gradient_penalty(D, real_samples, fake_samples, real_cond, fake_cond, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    alpha_squeezed = alpha.squeeze()
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # 混合条件
    mix_cond = alpha_squeezed.unsqueeze(-1) * real_cond + (1 - alpha_squeezed.unsqueeze(-1)) * fake_cond
    
    d_interpolates = D(interpolates, mix_cond)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradient_penalty =((gradients.norm(2, dim=(1,2)) - 1) ** 2).mean()

    return gradient_penalty


# 计算条件损失
def compute_condition_loss(fake_x, gen_cond):
    # 生成mask
    batch_size = fake_x.size(0)
    device = fake_x.device
    tox_mask = torch.zeros(batch_size, 1022, device=device)
    anti_mask = torch.zeros(batch_size, 1022, device=device)
    for i in range(batch_size):
        tox_len = int(gen_cond[i,0].item())
        anti_len = int(gen_cond[i,1].item())
        tox_mask[i, :tox_len] = 1
        anti_mask[i, :anti_len] = 1
    
    # 计算有效区域外的特征应接近零
    anti_features = fake_x[:, :, :1280]
    tox_features = fake_x[:, :, 1280:]
    
    # 抗毒素区域外的损失
    anti_out = anti_features * (1 - anti_mask.unsqueeze(-1))
    anti_loss = (anti_out ** 2).mean()
    
    # 毒素区域外的损失
    tox_out = tox_features * (1 - tox_mask.unsqueeze(-1))
    tox_loss = (tox_out ** 2).mean()
    
    return anti_loss + tox_loss


def sample_conditions(toxin_kde, antitoxin_kde, num_samples, device):
    tox_len = toxin_kde.sample(num_samples).squeeze(1).clip(1, 1022).astype(int)
    anti_len = antitoxin_kde.sample(num_samples).squeeze(1).clip(1, 1022).astype(int)
    gen_cond = torch.tensor(
        np.stack([tox_len, anti_len], axis=1),
        dtype=torch.float32
    ).to(device)

    return gen_cond, tox_len, anti_len


# 自适应损失权重调整（条件损失）
def adaptive_weight_cond_loss(mmd_value , target_range=(0.4, 0.2)):
    """mmd_value在0.1-0.3时保持权重，超出范围动态调整"""
    if mmd_value > target_range[0]:
        return 1.5  # 增强条件损失约束
    elif mmd_value < target_range[1]:
        return 0.7  # 减弱条件损失约束
    else:
        return 1.0


# 自适应损失权重调整（多样性损失）
def adaptive_weight_div_loss(diversity_score, target_range=(0.2, 0.4)):
    """当div_score在0.3-0.6时保持权重，超出范围动态调整"""
    if diversity_score < target_range[0]:
        return 1.5  # 增强多样性优化
    elif diversity_score > target_range[1]:
        return 0.7  # 减弱多样性优化
    else:
        return 1.0


# 训练设置
def train_gan(real_data, toxin_lengths, antitoxin_lengths, output_dir, 
             seed=42, epochs=100, batch_size=32, lr=2e-4, beta1=0.0, beta2=0.9,
             early_stop_patience=20, eval_interval=10, cond_coef=0.2, div_coef=0.2):

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # 初始化权重
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # 优化器
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=lr * 2, betas=(beta1, beta2))
    #schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=epochs)
    #schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=epochs)

    # 保存原始长度分布
    toxin_kde = KernelDensity(kernel='gaussian').fit(np.array(toxin_lengths).reshape(-1,1))
    antitoxin_kde = KernelDensity(kernel='gaussian').fit(np.array(antitoxin_lengths).reshape(-1,1))

    # 数据加载
    train_dataset = TADataset(real_data, toxin_lengths, antitoxin_lengths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        generator=torch.Generator().manual_seed(seed)  # 确定性shuffle
    )
    
    # 预处理所有真实数据的mask版本
    real_data_masked = []
    for x, tox_mask, antitox_mask in train_dataset:
        antitox_part = torch.tensor(x[:, :1280]) * antitox_mask.unsqueeze(-1)
        tox_part = torch.tensor(x[:, 1280:]) * tox_mask.unsqueeze(-1)
        real_data_masked.append(torch.cat([antitox_part, tox_part], dim=-1))
    real_data_masked = torch.stack(real_data_masked).to(device)
    real_data = real_data_masked

    # 初始化随机数生成器（用于生成噪声）
    z_rng = torch.Generator(device=device).manual_seed(seed)

    # 训练记录
    best_mmd = float('inf')
    best_acc_diff = 0.5
    best_epoch = 0
    mmd = 1.0
    mmd_threshold = 0.1
    wasserstein_dist_threshold = 2.0
    diversity_threshold = 0.3
    no_improve_epochs = 0
    history = {
        'd_loss': [], 'g_loss': [], 
        'wasserstein_dist': [],
        'mmd': [], 'diversity': [], 'c2st': []
    }

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        netG.train()
        netD.train()
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_wasserstein_dist = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for real_x, real_toxin_mask, real_antitoxin_mask in progress:
            real_x = real_x.to(device)
            real_toxin_mask = real_toxin_mask.to(device)
            real_antitoxin_mask = real_antitoxin_mask.to(device)
            batch_size = real_x.size(0)
            
            # 从真实数据获取条件
            real_tox_len_batch = real_toxin_mask.sum(dim=1).cpu().numpy()
            real_anti_len_batch = real_antitoxin_mask.sum(dim=1).cpu().numpy()
            real_cond = torch.tensor(np.stack([real_tox_len_batch, real_anti_len_batch], axis=1), 
                              dtype=torch.float32).to(device)

            # === 训练判别器 ===
            netD.zero_grad()
            
            # 真实样本
            d_real = netD(real_x, real_cond)
            loss_real = -d_real.mean()

            # 生成样本
            z = torch.randn(batch_size, 100, 
                           generator=z_rng,  # 使用固定种子的生成器
                           device=device)
            # 从KDE采样条件
            gen_tox_len = toxin_kde.sample(batch_size).squeeze(1).clip(1,1022)
            gen_anti_len = antitoxin_kde.sample(batch_size).squeeze(1).clip(1,1022)
            gen_cond = torch.tensor(np.stack([gen_tox_len, gen_anti_len], axis=1), 
                                   dtype=torch.float32).to(device)
            fake_x = netG(z, gen_cond)
            d_fake = netD(fake_x.detach(), gen_cond)
            loss_fake = d_fake.mean()

            # 梯度惩罚
            gp_weight = 10.0
            gp = compute_gradient_penalty(
                netD, real_x, fake_x, 
                real_cond, gen_cond, device
            )
            
            d_loss = loss_real + loss_fake + gp_weight * gp
            d_loss.backward()
            optD.step()

            # === 训练生成器 ===
            netG.zero_grad()
            fake_x = netG(z, gen_cond)
            d_fake = netD(fake_x, gen_cond)
            g_loss_adv = -d_fake.mean()

            # 条件一致性损失和多样性损失
            cond_loss = compute_condition_loss(fake_x, gen_cond)
            div_score = compute_diversity(fake_x, gen_tox_len, gen_anti_len)
            div_loss = 1.0 - div_score
            weight_div_loss = adaptive_weight_div_loss(div_score)
            weight_cond_loss = adaptive_weight_cond_loss(mmd)

            g_loss = g_loss_adv + weight_cond_loss * cond_coef * cond_loss + weight_div_loss * div_coef * div_loss
            g_loss.backward()
            optG.step()
            
            wasserstein_dist = d_real.mean().item() - d_fake.mean().item()
            
            # 累加 loss
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_wasserstein_dist += wasserstein_dist

            del fake_x, d_fake, z

            # tqdm 进度
            progress.set_postfix({"D Loss": d_loss.item(), "G Loss": g_loss.item(), "wasserstein_dist": wasserstein_dist})
        
        #schedulerG.step()
        #schedulerD.step()

        # ======= 指标计算 =======        
        # 计算平均损失（添加在epoch循环结束处）
        epoch_d_loss /= len(train_loader)
        epoch_g_loss /= len(train_loader)
        epoch_wasserstein_dist /= len(train_loader)

        # 保存历史记录
        history['d_loss'].append(epoch_d_loss)
        history['g_loss'].append(epoch_g_loss)
        history['wasserstein_dist'].append(epoch_wasserstein_dist)

        # ======= 定期评估 =======
        if (epoch + 1) % eval_interval == 0:
            netG.eval()
            with torch.no_grad():         
                # 生成评估样本
                z = torch.randn(len(real_data), 100, generator=z_rng, device=device)
                gen_cond, gen_tox_len, gen_anti_len = sample_conditions(toxin_kde, antitoxin_kde, len(real_data), device)
                fake_x = netG(z, gen_cond)
                
                # 计算MMD、c2st和多样性
                mmd = compute_mmd(real_data, fake_x, toxin_lengths, antitoxin_lengths, gen_tox_len, gen_anti_len)
                c2st_acc = compute_c2st(real_data, fake_x, toxin_lengths, antitoxin_lengths, gen_tox_len, gen_anti_len)
                diversity = compute_diversity(fake_x, gen_tox_len, gen_anti_len)
                history['mmd'].append(mmd)
                history['c2st'].append(c2st_acc)
                history['diversity'].append(diversity)
        
                # 保存最佳模型
                #if mmd < best_mmd and epoch_wasserstein_dist > wasserstein_dist_threshold and diversity > diversity_threshold:
                diff_accuracy_05 = abs(c2st_acc - 0.5)
                if diff_accuracy_05 < best_acc_diff:
                    best_acc_diff = diff_accuracy_05
                    #best_mmd = mmd
                    best_epoch = epoch + 1
                    no_improve_epochs = 0

                    model_path = os.path.join(output_dir, f"best_model_epoch{epoch + 1}.pth")
                    torch.save({                        
                        'generator': netG.module.state_dict() if isinstance(netG, nn.DataParallel) else netG.state_dict(),
                        'gen_anti_lens': gen_anti_len,
                        'gen_tox_lens': gen_tox_len,
                        'real_anti_lens': antitoxin_lengths,
                        'real_tox_lens': toxin_lengths,
                        'discriminator': netD.module.state_dict() if isinstance(netD, nn.DataParallel) else netD.state_dict(),
                        'generate_sample': fake_x,
                        'epoch': epoch + 1,
                        'mmd': mmd,
                        'c2st_acc': c2st_acc,
                        'diversity': diversity
                    }, model_path)
                else:
                    no_improve_epochs += eval_interval
                
                # 定期保存
                save_interval = 100
                if (epoch + 1) % save_interval == 0 or epoch == 0:
                    model_path = os.path.join(output_dir, f"best_model_epoch{epoch + 1}.pth")
                    torch.save({                        
                        'generator': netG.module.state_dict() if isinstance(netG, nn.DataParallel) else netG.state_dict(),
                        'gen_anti_lens': gen_anti_len,
                        'gen_tox_lens': gen_tox_len,
                        'real_anti_lens': antitoxin_lengths,
                        'real_tox_lens': toxin_lengths,
                        'discriminator': netD.module.state_dict() if isinstance(netD, nn.DataParallel) else netD.state_dict(),
                        'generate_sample': fake_x,
                        'epoch': epoch + 1,
                        'mmd': mmd,
                        'c2st_acc': c2st_acc,
                        'diversity': diversity
                    }, model_path)
                    
                # 清理变量和缓存
                del fake_x
                torch.cuda.empty_cache()

                # 打印指标
                print(f"Epoch {epoch + 1} | C2ST Acc: {c2st_acc:.3f} | MMD: {mmd:.4f} | WD-Dist: {epoch_wasserstein_dist:.3f} | Diversity: {diversity:.3f} | G lr: {optG.param_groups[0]['lr']:.2e} | D lr: {optD.param_groups[0]['lr']:.2e}")
                print(f"D Loss: {epoch_d_loss:.4f} | G Loss: {epoch_g_loss:.4f} | weight_div_loss: {weight_div_loss:.3f} | weight_cond_loss: {weight_cond_loss:.3f}")
                
                # 早停判断
                """
                stop_condition = (
                    (mmd < mmd_threshold) and 
                    (no_improve_epochs >= early_stop_patience) and 
                    (epoch_wasserstein_dist > wasserstein_dist_threshold) and
                    (diversity > diversity_threshold)
                )
                """
                stop_condition = (
                    (no_improve_epochs >= early_stop_patience) and 
                    (0.45 < c2st_acc < 0.55)
                )
                if stop_condition:
                    print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}")
                    return history, netG, netD

    return history, netG, netD


def generate_samples(checkpoint_path, num_samples, seed, output_dir, device):
    # 加载模型
    checkpoint = torch.load(checkpoint_path)
    netG = Generator().to(device)

    # 处理多GPU参数名称
    state_dict = checkpoint['generator']

    netG.load_state_dict(state_dict)
    
    # 生成随机长度
    toxin_kde = checkpoint['toxin_kde']
    antitoxin_kde = checkpoint['antitoxin_kde']
    toxin_lengths = toxin_kde.sample(num_samples).clip(1,1022).astype(int)
    antitoxin_lengths = antitoxin_kde.sample(num_samples).clip(1,1022).astype(int)

    # 从KDE采样条件
    cond = torch.tensor(np.stack([toxin_lengths, antitoxin_lengths], axis=1), 
                       dtype=torch.float32).to(device)
    
    # 生成噪声
    z_rng = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(num_samples, 100, generator=z_rng, device=device)

    # 生成样本
    with torch.no_grad():
        samples = netG(z, cond)
    
    # 保存结果
    output_path = os.path.join(output_dir, f"generated_samples_seed{seed}.pt")
    torch.save({
        'samples': samples.cpu(),
        'antitoxin_lengths': antitoxin_lengths,
        'toxin_lengths': toxin_lengths,
        'generator_seed': seed
    }, output_path)
    print(f"[Success] Generated samples saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="TA System GAN Training/Generation")
    
    # 通用参数
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate'],
                       help='Run mode: train or\ generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Compute device (default: cuda if available)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (train mode only)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (train mode only)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (train mode only)')
    parser.add_argument('--early_stop', type=int, default=20,
                       help='Early stopping patience (train mode only)')
    parser.add_argument('--eval_interval', type=int, default=10,
                       help='Evaluation interval in epochs (train mode only)')
    parser.add_argument('--data_path', type=str,
                       help='Path to training data numpy file (train mode only)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for models/samples')
    
    # 生成参数
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model checkpoint (generate mode only)')
    parser.add_argument('--num_samples', type=int,
                       help='Number of samples to generate (generate mode only)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.mode == 'train':
        # 加载数据
        print("\n=== Initializing Training ===")
        real_samples = np.load(args.data_path)
        data = pd.read_table('/public/guanjh/TA/TAfinder_2.0/dataset/data/pos_sample.tsv', header=None)
        antitoxin_lengths = data[3].apply(len)
        toxin_lengths = data[4].apply(len)
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Print configuration
        print("\nTraining Configuration:")
        print(f"| {'Parameter':<15} | {'Value':<12} |")
        print("|-----------------|--------------|")
        print(f"| Random seed     | {args.seed:<12} |")
        print(f"| Training epochs | {args.epochs:<12} |")
        print(f"| Batch size      | {args.batch_size:<12} |")
        print(f"| Output directory| {args.output_dir:<12} |")
        print("\nStarting training...")

        # 运行训练
        history, netG, netD = train_gan(
            real_data=real_samples, 
            toxin_lengths=toxin_lengths, 
            antitoxin_lengths=antitoxin_lengths, 
            output_dir=args.output_dir,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            early_stop_patience=args.early_stop,
            eval_interval=args.eval_interval      
        )

        # 保存训练历史
        history_path = os.path.join(args.output_dir, 'training_history.pt')
        torch.save(history, history_path)
        print(f"\n[Success] Training history saved to: {history_path}")

    elif args.mode == 'generate':
        if not args.model_path:
            raise ValueError("Model path must be specified in generate mode")
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Print configuration
        print("\nGeneration Configuration:")
        print(f"| {'Parameter':<15} | {'Value':<12} |")
        print("|-----------------|-------------|")
        print(f"| Random seed     | {args.seed:<12} |")
        print(f"| Num samples     | {args.num_samples:<12} |")
        print(f"| Output directory| {args.output_dir:<12} |")
        print("\nStarting generation...")
        
        generate_samples(args.model_path, args.num_samples, args.seed, args.output_dir, args.device)

# 使用示例
if __name__ == "__main__":
    main()