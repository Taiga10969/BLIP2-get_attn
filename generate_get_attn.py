import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from lavis.models import load_model_and_preprocess

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
#raw_image = Image.open('your_path/BLIP2-get_attn/sample_img/merlion.png').convert('RGB')
raw_image = Image.open('/taiga/experiment/BLIP2-get_attn/sample_img/merlion.png').convert('RGB')
prompt = "Question: which city is this? Answer:"

# 画像の前処理
image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)

# generate推論 (num_beams=1にしないと，T5での推論回数が多くなり，attentionが多く出力される)
output = model.generate({"image": image, "prompt": prompt}, num_beams=1)

# 推論結果の確認
print("output['output_text'] : ", output['output_text'])  # ['singapore']
# サブワードの出力ベースの出力結果を確認
print("output['output_word'] : ", output['output_word'])  # ['<pad>', '▁sing', 'a', 'pore', '</s>'] *実際にモデルが出力した単語は'<pad>'以降

# Attention ========================================================================================
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

# show_attention ===========================================================================

# Vision_encoderのAttentio Weightの可視化（最終層のヘッド平均したものを可視化）
plt.imshow(np.mean(vision_encoder_attn[-1], axis=0))
#plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'attention/vision_encoder_attn_last_layer.png', transparent=True)
#plt.savefig(f'attention/vision_encoder_attn_last_layer.svg', transparent=True)
plt.close()

# Q-formerのcross-attentionのAttention Weightの可視化（最終層のヘッド平均したものを可視化）
plt.imshow(np.mean(qformer_cross_attn[-1], axis=0))
#plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'attention/qformer_cross_attn_last_layer.png', transparent=True)
#plt.savefig(f'attention/qformer_cross_attn_last_layer.svg', transparent=True)
plt.close()

# Q-formerのself-attentionのAttention Weightの可視化（最終層のヘッド平均したものを可視化）
plt.imshow(np.mean(qformer_self_attn[-1], axis=0))
#plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'attention/qformer_self_attn_last_layer.png', transparent=True)
#plt.savefig(f'attention/qformer_self_attn_last_layer.svg', transparent=True)
plt.close()

# T5 Decoderのcross-attentionのAttention Weightの可視化（最終層のヘッド平均したものを可視化）
plt.imshow(np.mean(t5decoder_cross_attn[-1], axis=0))
#plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.savefig(f'attention/t5decoder_cross_attn_last_layer.png', transparent=True)
#plt.savefig(f'attention/t5decoder_cross_attn_last_layer.svg', transparent=True)
plt.close()