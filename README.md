# BLIP2-get_attn
BLIP-2のモデルからの出力としてAttention Weightの情報を出力するように改変したプログラムになります．<br>

## outline
原プログラム：[LAVIS - A Library for Language-Vision Intelligence](https://github.com/salesforce/LAVIS)<br>

Attentionを取り出せるように変更したファイルは，以下のファイルになります．<br>
- lavis/models/eva_vit.py
- lavis/models/blip2_models/blip2_t5.py

*変更した箇所には，`# modi_ATTN` というコメントが追加されています．



## 使用方法
- repositryのclone
```
git clone https://github.com/Taiga10969/BLIP2-get_attn.git
```
- 必要ライブラリのインストール
```
cd BLIP2-get_attn
```
```
pip install -r requirements.txt
```
- generate()時のAttentionの取得コードの実行
```
python3 generate_get_attn.py
```

## Attention t5_Q-former_seed Rollout
単語毎に画像に対するAttentionを可視化するには，T5 DecoderのAttention WeightからQ-Formerを橋渡しとして画像エンコーダまで遡るようにしてAttentionを見なければならない．
この可視化方法では，t5のDecoder部分のCross Attention（ヘッド方向に平均をとり，層方向に乗算したもの）からQ-Formerからの出力（画像の特徴を32トークンに圧縮したもの）に対する注目箇所を確認することができる．
つまり，ある単語生成時におけるQ-Formerの32トークンの内どのトークンに注目しているかが獲得できる．したがって，Q-Former部分のCross Attention（ヘッド方向に平均をとり，層方向に乗算したもの）を32トークンの方向に，ある単語生成時におけるQ-Formerの32トークンの内どのトークンに注目度を基とした重み付き平均をとることによって，単語生成時における画像エンコーダからの257トークン（クラストークン1 + パッチトークン256）の特徴の内，どのトークンに注目しているのかを獲得できる．さらに，この注目度を利用して，Attention Rolloutした画像エンコーダのAttention Weightを257トークンの方向に重み付き平均をとることで，T5，Q-Formerの注目度を考慮した画像に対するAttentionを可視化している．
<be>
※現状の可視化方法では，単語毎に著しく画像の注目箇所が変化するような挙動は確認できていない．
![Test Image 3](https://github.com/Taiga10969/BLIP2-get_attn/blob/main/sample_img/BLIP2-Attention.png)

```
python3 seed_rollout.py
```

## TO-DO

- [x] Attentionを取り出せるようにプログラムを変更
- [ ] 詳細な説明を記述
- [ ] 可視化方法を模索
