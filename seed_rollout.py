import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from lavis.models import load_model_and_preprocess
from utils import Attention_Rollout, heatmap_to_rgb, normalize_attention_map

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')


# モデルの読み込み
model, vis_processors, _ = load_model_and_preprocess(
    name='blip2_t5',
    model_type='pretrain_flant5xl', #pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
    is_eval=True,
    device=device
)

# 入力データの用意
raw_image = Image.open('your_path/BLIP2-get_attn/sample_img/merlion.png').convert('RGB')
prompt = "Question: What is the building you see in the back and the object you see in front? Answer:"

# 画像の前処理
image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)

# generate推論 (num_beams=1にしないと，T5での推論回数が多くなり，attentionが多く出力される)
output = model.generate({"image": image, "prompt": prompt}, num_beams=1)

# 推論結果の確認
print("output['output_text'] : ", output['output_text'])  # ['singapore']
# サブワードの出力ベースの出力結果を確認
print("output['output_word'] : ", output['output_word'])  # ['<pad>', '▁sing', 'a', 'pore', '</s>'] *実際にモデルが出力した単語は'<pad>'以降

# Attention << seed_rollout >> ===============================================================================
batch = 0 # batchの画像に対してのAttentionを収集する

# Vision_encoderのAttention
vision_encoder_attn = np.array(output['vision_attn']) # output['vision_attn']=(39, 1, 16, 257, 257) >> (layer_depth, batch, head, height, weidth)
vision_encoder_attn = vision_encoder_attn[:, batch, :,:,:]
print("np.shape(vision_encoder_attn)", np.shape(vision_encoder_attn))  # (39, 16, 257, 257) >> (layer_depth, head, height, weidth)


# Q-formerのAttention
# output['Qformer_attn']では，Self-AttentionとCross-Attentionが交互に適応されており，Cross-Attentioは2層に1回組み込まれている．
# その為，Self-AttentionとCross-Attentionに分ける作業を行う必要がある．
qformer_cross_attn = []
qformer_self_attn = []

for layer in range(0,len(output['Qformer_attns']),2):
    qformer_cross_attn.append(output['Qformer_attns'][layer][batch].to(torch.float).cpu().detach().numpy())     # torch.Size([1, 12, 32, 257]) >> [batch, head, query, patchs]
    qformer_self_attn.append(output['Qformer_attns'][layer+1][0][batch].to(torch.float).cpu().detach().numpy()) # torch.Size([1, 12, 32, 64]) >> [batch, head, query, query+(shared)]
    qformer_self_attn.append(output['Qformer_attns'][layer+1][1][batch].to(torch.float).cpu().detach().numpy()) # torch.Size([1, 12, 32, 64]) >> [batch, head, query, query+(shared)]

print("np.shape(qformer_cross_attn) : ", np.shape(qformer_cross_attn)) # (6, 12, 32, 257) >> (layer, head, query, patchs)
print("np.shape(qformer_self_attn) : ", np.shape(qformer_self_attn))   # (12, 12, 32, 64) >> (layer, head, query, query+(shared))


# T5のAttention (最後の単語を出力した際のAttentionを取得)
t5decoder_cross_attn = []
for layer in range(0,len(output['t5_Xattn'][-1]),1): # [-1]で最終単語出力時のAttentionを取得
    t5decoder_cross_attn.append(output['t5_Xattn'][-1][layer][batch].to(torch.float).cpu().detach().numpy())

print("np.shape(t5decoder_cross_attn) : ", np.shape(t5decoder_cross_attn))   # (24, 32, 4, 42) >> (layer, head, word, dim)

# << seed_rollout >> ============================================================================================

word = 2

t5_attention = np.prod(np.mean(t5decoder_cross_attn, axis=1), axis=0)
print('np.shape(t5_attention) : ', np.shape(t5_attention[word,:32]))

plt.imshow(t5_attention[:,:32])
#plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'attention/layer_prod_head_mean_t5_cross_attn.png', transparent=True)
#plt.savefig(f'attention/layer_prod_head_mean_t5_cross_attn.svg', transparent=True)
plt.close()

qformer_attention = np.prod(np.mean(qformer_cross_attn, axis=1), axis=0)
print('np.shape(qformer_attention) : ', np.shape(qformer_attention))

plt.imshow(qformer_attention)
#plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'attention/layer_prod_head_mean_qformer_cross_attn.png', transparent=True)
#plt.savefig(f'attention/layer_prod_head_mean_qformer_cross_attn.svg', transparent=True)
plt.close()


# t5 corss attentionの値を用いて，qformerのattentionの重み付き平均を取得する
t5_attention_related_qformer = t5_attention[word,:32]
t5_attention_related_qformer_normalized = t5_attention_related_qformer / np.sum(t5_attention_related_qformer) # t5_attentionを正規化する
qformer_weighted_avg = np.dot(t5_attention_related_qformer_normalized, qformer_attention)
print('np.shape(qformer_weighted_avg) : ', np.shape(qformer_weighted_avg))

qformer_weighted_avg = qformer_weighted_avg / np.sum(qformer_weighted_avg)

# vision rollout

vision_attn = Attention_Rollout(vision_encoder_attn)
print('np.shape(vision_attn) : ', np.shape(vision_attn))

vision_attn = np.dot(np.transpose(qformer_weighted_avg), vision_attn)
print('np.shape(vision_attn) : ', np.shape(vision_attn))
attention = np.reshape(vision_attn[1:],(16,16))

plt.imshow(attention)
#plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'attention/attention_word_test.png', transparent=True)
#plt.savefig(f'attention/attention_word_test.svg', transparent=True)
plt.close()




img_ = np.transpose(image.cpu().numpy()[0], (1,2,0))
normalized_data = (img_ - img_.min()) / (img_.max() - img_.min())
img_ = (normalized_data * 255).astype(np.uint8)
normalization_method = 'min-max'

for word in range(len(output['output_word'])-1):
    t5_attention = np.prod(np.mean(t5decoder_cross_attn, axis=1), axis=0)

    qformer_attention = np.prod(np.mean(qformer_cross_attn, axis=1), axis=0)

    # t5 corss attentionの値を用いて，qformerのattentionの重み付き平均を取得する
    t5_attention_related_qformer = t5_attention[word,:32]
    t5_attention_related_qformer_normalized = t5_attention_related_qformer / np.sum(t5_attention_related_qformer) # t5_attentionを正規化する
    qformer_weighted_avg = np.dot(t5_attention_related_qformer_normalized, qformer_attention)

    qformer_weighted_avg = qformer_weighted_avg / np.sum(qformer_weighted_avg)

    # vision rollout

    vision_attn = Attention_Rollout(vision_encoder_attn)

    vision_attn = np.dot(np.transpose(qformer_weighted_avg), vision_attn)
    #vision_attn = vision_attn/np.sum(vision_attn)
    attention = np.reshape(vision_attn[1:],(16,16))

    plt.imshow(attention)
    plt.title(output['output_word'][word+1])
    #plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.savefig(f'attention/attention_word{word}.png', transparent=True)
    #plt.savefig(f'attention/attention_word{word}.svg', transparent=True)
    plt.close()

    
    attention_map_ = cv2.resize(attention, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    
    attention_map_rgb = heatmap_to_rgb(attention_map_)
    attention_map = cv2.addWeighted(img_, 0.5, attention_map_rgb, 0.5, 0)

    plt.imshow(attention_map)
    plt.title(output['output_word'][word+1])
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.savefig(f'attention/AttentionMap_word{word}.png', transparent=True)
    #plt.savefig(f'attention/AttentionMap_word{word}.svg', transparent=True)
    plt.close()
