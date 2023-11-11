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
