import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap


def get_embedding_GAN(file):
    GAN_embedding = pd.read_csv(file, sep=',', header=None).values[:,:-1].tolist()
    return(np.array(GAN_embedding))


def get_embedding(model, iterator, device):
    embeddings = []

    model.eval()

    with torch.no_grad():
        for labels, strs, toks in iterator:            
            strs_1, strs_2 = zip(*strs)
            toks_1, toks_2 = zip(*toks)
            toks_1 = torch.stack(toks_1).to(device)
            toks_2 = torch.stack(toks_2).to(device)
            emb = model(strs_1, strs_2, toks_1, toks_2)
            embeddings.append(emb)
    
    embeddings = torch.cat(embeddings).cpu().numpy()
    return embeddings


def get_embedding_single(model, iterator, device):
    embeddings = []

    model.eval()

    with torch.no_grad():
        for labels, strs, toks in iterator:
            toks = toks.to(device)
            emb = model(strs, toks)
            embeddings.append(emb)
    
    embeddings = torch.cat(embeddings).cpu().numpy()
    return embeddings


def plot_tsne(x, y, color_dict, title, outdir):    
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x)
    
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)

    plt.figure(figsize=(4, 4))

    for i, (name, color) in enumerate(color_dict.items()):
        plt.scatter(x_norm[y == i, 0], x_norm[y == i, 1], c=color, s=10, label=name, alpha = 0.8)
 
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")

    plt.title(title)
    plt.legend(loc="best", fontsize=10)
    plt.savefig(os.path.join(outdir, "embedding_tsne.pdf"), dpi=300)


def plot_umap(x, y, color_dict, title, outdir):    
    my_umap = umap.UMAP(n_components=2, random_state=42)
    x_umap = my_umap.fit_transform(x)
    
    x_min, x_max = x_umap.min(0), x_umap.max(0)
    x_norm = (x_umap - x_min) / (x_max - x_min)

    plt.figure(figsize=(4, 4))

    for i, (name, color) in enumerate(color_dict.items()):
        plt.scatter(x_norm[y == i, 0], x_norm[y == i, 1], c=color, s=10, label=name, alpha=0.8)
 
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    plt.title(title)
    plt.legend(loc="best", fontsize=10)
    plt.savefig(os.path.join(outdir, "embedding_umap.pdf"), dpi=300)


def plot_embedding(model, iterator, outdir, device, datadir, GAN=False, single=False, keep_seq_len=False):
    if not single:
        test_data = os.path.join(datadir, "test.tsv")
        df_data = pd.read_csv(test_data, header = None, sep = '\t')
        #df_data.loc[df_data[2] == 2, 2] = 0 # 如果不需要对label 2标注颜色，就加这一行
        df_data.loc[df_data[2] == 3, 2] = 2
        truth = np.array(df_data[2].tolist())    

        embeddings = get_embedding(model, iterator, device)
        if GAN:
            GAN_embedding = get_embedding_GAN('/public/guanjh/TA/TAfinder_2.0/dataset/GAN_result_FFPred/Iteration_13000.txt')
            embeddings = np.concatenate([embeddings, GAN_embedding], axis=0)

            if 3 in truth:
                truth = np.append(truth, np.full(GAN_embedding.shape[0], 4))
                color_dict = {'Non-TA pair':'#dddddd', 'TA pair': '#FB8072', 'Non-T & AT/Non-AT & T': '#FFB90F', 'TA pair in reverse order': '#80B1D3', 'GAN-generated': '#8DD3C7'}
            elif 2 in truth:
                truth = np.append(truth, np.full(GAN_embedding.shape[0], 3))
                color_dict = {'Non-TA pair':'#dddddd', 'TA pair': '#FB8072', 'Non-T & AT/non-AT & T': '#FFB90F', 'GAN-generated': '#8DD3C7'}
            else:
                truth = np.append(truth, np.full(GAN_embedding.shape[0], 2))
                color_dict = {'Non-TA pair':'#dddddd', 'TA pair': '#FB8072', 'GAN-generated': '#8DD3C7'}

        else:
            if 3 in truth:
                color_dict = {'Non-TA pair':'#e6e6e6', 'TA pair': '#FB8072', 'Non-T & AT/Non-AT & T': '#FFB90F', 'TA pair in reverse order': '#80B1D3', }
            elif 2 in truth:
                #color_dict = {'Non-TA pair':'#dddddd', 'TA pair': '#FB8072', 'Non-T & AT/non-AT & T': '#FFB90F'}
                color_dict = {'Non-TA pair':'#e6e6e6', 'TA pair': '#FB8072', 'TA pair in reverse order': '#80B1D3'}
            else:
                color_dict = {'Non-TA pair':'#e6e6e6', 'TA pair': '#FB8072'}
                #color_dict = {'Reverse TA pair': '#eac162', 'Canonical TA pair': '#9a91d4'}
        
        #title = "T-SNE"
        title = "UMAP projection of sequence embedding"
        plot_tsne(embeddings, truth, color_dict, "T-SNE projection of sequence embedding", outdir)
        plot_umap(embeddings, truth, color_dict, "UMAP projection of sequence embedding", outdir)

    else:
        test_data = os.path.join(datadir, "test.tsv")
        df_data = pd.read_csv(test_data, header = None, sep = '\t')
        truth = np.array(df_data[1].tolist())

        embeddings = get_embedding_single(model, iterator, device)
        if GAN:
            pass
        else:
            color_dict = {'Neg':'#dddddd', 'Pos': '#80B1D3'}
        
        plot_tsne(embeddings, truth, color_dict, "T-SNE projection of sequence embedding", outdir)
        plot_umap(embeddings, truth, color_dict, "UMAP projection of sequence embedding", outdir)
    

