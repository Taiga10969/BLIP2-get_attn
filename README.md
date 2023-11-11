# BLIP2-get_attn
BLIP-2のモデルからの出力としてAttention Weightの情報を出力するように改変したプログラムになります．<br>
原プログラム：[LAVIS - A Library for Language-Vision Intelligence](https://github.com/salesforce/LAVIS)




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
