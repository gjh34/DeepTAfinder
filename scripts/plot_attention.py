import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time
import os
import sys
import logomaker as lm

attn_dir = sys.argv[1]
data_dir = sys.argv[2]

attn_dict = np.load(os.path.join(attn_dir, 'attn.npz'))
data = pd.read_csv(os.path.join(data_dir, 'test.tsv'), header = None, sep = '\t')

offset = .1
for key, value in attn_dict.items():
    attn = value
    max_attn = max(attn)
    min_attn = min(attn)
    key_split = key.split('_')
    label = key_split[0]

    # 确定原始序列和序列名称
    protein_type = key_split[1]    
    if protein_type == 'AT':
        seq_name = data.loc[data[2] == int(label), 0].values[0]
        seq = data.loc[data[2] == int(label), 3].values[0]
    elif protein_type == 'T':
        seq_name = data.loc[data[2] == int(label), 1].values[0]
        seq = data.loc[data[2] == int(label), 4].values[0]
    else:
        print("No match found.")

    # 画 logo 图
    df = pd.DataFrame({'character': list(seq), 'value': attn})
    saliency_df = lm.saliency_to_matrix(
        seq = df['character'], values = (df['value'] - min_attn) / (max_attn - min_attn) + offset)
    rows = math.ceil(len(saliency_df) / 100)
    labels = [0.00, 0.25, 0.50, 0.75, 1.00]
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2*rows))

    for i in range(rows):
        tempdf = saliency_df[i*100:(i+1)*100]
        tempdf.set_index(
            [pd.Index(np.array(tempdf.index)-100*i)], inplace=True)
        logo = lm.Logo(tempdf,
                       color_scheme='skylign_protein',
                       vpad=0,
                       width=0.8,
                       font_weight='normal',
                       ax=axes if rows == 1 else axes[i])

        logo.ax.set_ylim([0, 1])
        logo.ax.set_yticks(np.array(labels) + offset)
        logo.ax.set_yticklabels(labels)
        logo.ax.set_xlim([-1, 100])
        logo.ax.set_xticks([19, 39, 59, 79, 99])
        logo.ax.set_xticklabels(np.array([20, 40, 60, 80, 100]) + 100*i)
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left'], visible=True)
        logo.ax.axhline(offset, color='gray', linewidth=1, linestyle='--')

    df.to_csv(os.path.join(attn_dir, f'attn_{seq_name}.csv'), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(attn_dir, f'attn_{seq_name}.pdf'), dpi=300)
