import numpy as np
import matplotlib.pyplot as plt

def Attention_Rollout(vision_attns):
    mean_head = np.mean(vision_attns, axis=1)
    mean_head = mean_head + np.eye(mean_head.shape[1])
    mean_head = mean_head / mean_head.sum(axis=(1,2))[:, np.newaxis, np.newaxis]

    v = mean_head[-1]
    for n in range(1,len(mean_head)):
        v = np.matmul(v, mean_head[-1-n])
    
    return v

#def heatmap_to_rgb(heatmap):
#    # カラーマップに変換
#    colormap = plt.get_cmap('jet')
#    colored_heatmap = colormap(heatmap)
#    # RGBに変換
#    rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
#    return rgb_image

def heatmap_to_rgb(heatmap, cmap='jet'):
    # 0から1の範囲に正規化
    normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    # カラーマップに変換
    colormap = plt.get_cmap(cmap)
    colored_heatmap = (colormap(normalized_heatmap) * 255).astype(np.uint8)
    # RGBに変換
    rgb_image = colored_heatmap[:, :, :3]
    
    return rgb_image

def normalize_attention_map(attention_map, normalization_method='min-max'):
    if normalization_method == 'min-max':
        min_attention = attention_map.min()
        max_attention = attention_map.max()
        normalized_attention = (attention_map - min_attention) / (max_attention - min_attention)
    elif normalization_method == 'max':
        normalized_attention = attention_map / attention_map.max()
    elif normalization_method == 'z-score':
        mean_attention = attention_map.mean()
        std_attention = attention_map.std()
        normalized_attention = (attention_map - mean_attention) / std_attention
    elif normalization_method == 'softmax':
        exp_attention = np.exp(attention_map)
        normalized_attention = exp_attention / np.sum(exp_attention)
    else:
        raise ValueError("Invalid normalization method")

    return normalized_attention
