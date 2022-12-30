#!/usr/bin/env python
# coding: utf-8

# # Hugging Face Library 'Transformer'およびT5Tokenizerのダウンロード
# 
# 参考(https://qiita.com/takubb/items/fd972f0ac3dba909c293)これを基に改造し、最新のGoogle Colaboratoryで動作するようにした

# In[1]:


import argparse

parser = argparse.ArgumentParser(description='liBERTy testbed')
tp = lambda x:list(map(str, x.split(',')))
parser.add_argument('-l', '--num_of_learn', type=int, default=100,
    help='number (default 100) of learn, which determins the rate of dataset for the use of learning.')
parser.add_argument('-v', '--num_of_validation', type=int, default=100,
    help='number of validation to shorten the validation time.')
parser.add_argument('-e', '--max_epoch', type=int, default=20, 
    help='number (default 20) of epoch to be executed for learning loop')
parser.add_argument('-b', '--batch_size', type=int, default=64, 
    help='size (defaualt 64) of batch for learning process')
parser.add_argument('-a', '--article_type', type=int, default=0, choices=[0,1], 
    help='article type (0: dokujo_it=default, 1:dokujo_peachy')
parser.add_argument('-t', '--transformflags', type=tp, default = ['n'], #default=['r','i','d','s'], 
    help='NLP-JP transformer (default n) r:synreplace i:randinsert d:randdelete s:randswap n:none')
parser.add_argument('-r', '--synreplace_rate', type=int, default=1, 
    help='rate (default 1) of synreplace_rate par sentence as int for transformers.')
parser.add_argument('-i', '--randinsert_rate', type=int, default=3, 
    help='rate (default 3) of randinsert of dataset par sentence as int for transformers.')
parser.add_argument('-d', '--randdelete_rate', type=float, default=0.15, 
    help='probability (default 0.15) of lranddelete in a sentence as float of dataset for transformers.')
parser.add_argument('-s', '--randswap_rate', type=int, default=2, 
    help='rate (default 2) of randswap of dataset per sentence as int for transformers.')
parser.add_argument('-f', '--jupyter', default='CMD', 
    help='executed from jupyter')
args = parser.parse_args()

if args.jupyter == 'CMD':
    numof_learn = args.num_of_learn
    numof_validation = args.num_of_validation
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    transformflags = args.transformflags
    synreplace_rate = args.synreplace_rate
    randinsert_rate = args.randinsert_rate
    randdelete_rate = args.randdelete_rate
    randswap_rate = args.randswap_rate
    articletype = args.article_type
else:
    numof_learn = 100
    numof_validation = 200
    max_epoch = 20
    batch_size = 64
    transformflags = ['n'] #['r','i','d','s']
    synreplace_rate = 1
    randinsert_rate = 3
    randdelete_rate = 0.15
    randswap_rate = 2   
    articletype = 0
articlelabel = ['dokujo_it', 'dokujo_peachy']
print("num_of_learn:",numof_learn," max_epoch:", max_epoch," num_of_batch:", batch_size,
      " articletype:", articlelabel[articletype])
filestr = "l:"+str(numof_learn)+"_e:"+str(max_epoch)+"_b:"+str(batch_size)+"_t:"+''.join(transformflags)+    "_r:"+str(synreplace_rate)+'_i:'+str(randinsert_rate)+'_d:'+str(randdelete_rate)+'_s:'+str(randswap_rate)+    "_a:"+articlelabel[articletype]
print(filestr)


# In[2]:


#!export CUDA_LAUNCH_BLOCKING=1
# !pip install torch
#!pip install torchvision
#!pip install transformers
#!apt install swig
# Sentencepieceのインストール
#!pip install sentencepiece
#!pip install mecab-python3


# In[3]:


import os
import random
import re
import csv
import glob
import torchvision
import statistics
import numpy as np
import MeCab
import copy

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings

from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading


# # PyTorchとGPU設定

# In[4]:


#!pip install torch
import torch
# GPUが使えれば利用する設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# # Data Augmentation kansuu

# In[5]:


from transformers import RobertaForMaskedLM
robertamodel = RobertaForMaskedLM.from_pretrained("rinna/japanese-roberta-base")


# In[6]:


# synreplace - replace kasho kosuu
# randinsert - tasu kotoba no kazu
# randdelete - delete kakuritsu
# randswap - swap kaisuu

class synreplace(object):
    def __init__(self, num):
        self.num = num
    def __call__(self, textlist):
        # textlist: honbun no list
        textlen = torch.where(textlist == 3)[0][0]
        for n in range(self.num):
            # chikan shiro
            masked_idx = random.randint(2, textlen-1)
            textlist[masked_idx] = 6
            # convert to tensor
            token_tensor = torch.tensor(textlist)
            # get the top 10 predictions of the masked token
            self.model = robertamodel.eval()
            with torch.no_grad():
                outputs = self.model(torch.unsqueeze(token_tensor, 0))
                predictions = outputs[0][0, masked_idx].topk(1)
            for i, index_t in enumerate(predictions.indices):
                index = index_t.item()
            textlist[masked_idx] = index
        return textlist

class randinsert(object):
    def __init__(self, num):
        self.num = num
    def __call__(self, textlist):
        for n in range(self.num):
            insword = textlist[random.randint(1,len(textlist)-1)]
            i = random.randint(1,len(textlist)-1)
#            print('len: ', len(textlist))
#            print(i)
            while textlist[i] == 3:
                i = random.randint(1,len(textlist)-1)
#                print(i)
            textlist = torch.cat([textlist[0:i], torch.tensor([insword]), textlist[i:-1]])
        return textlist

class randdelete(object):
    def __init__(self, num):
        self.num = num
    def __call__(self, textlist):
#        print(textlist.shape)
        for i in range(3,len(textlist)-1):
            if textlist[i] == 3:
                continue
            r = random.uniform(0, 1)
            if r < self.num:
#                textlist.pop(i)
                textlist = torch.cat([textlist[0:i], textlist[i+1:], torch.tensor([3])])
#                print(textlist)
        return textlist

class randswap(object):
    def __init__(self, num):
        self.num = num
    def __call__(self, textlist):
        counter = 0
        #rs_sents = np.zeros(len(textlist), dtype=object)
        for i in range(len(textlist)):
            while self.num > counter:
                box = 0
                random_idx_1 = random.randint(1, len(textlist)-1)
                while textlist[random_idx_1] == 3:
                    random_idx_1 = random.randint(0, len(textlist)-1)
                random_idx_2 = random.randint(1, len(textlist)-1)
                while random_idx_1 == random_idx_2 or textlist[random_idx_2] == 3:
                    random_idx_2 = random.randint(0, len(textlist)-1)
                    # print(random_idx_1, random_idx_2)
                box = textlist[random_idx_1]
                textlist[random_idx_1] = textlist[random_idx_2]
                textlist[random_idx_2] = box
                counter += 1
        return textlist


# # Custom Tensor Dataset
# https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset

# # データセットの準備

# ライブドアニュースコーパスをダウンロード
# 
#     ダウンロードしたファイルは圧縮（tar.gz形式）ファイル
#     様々なジャンル（IT,スポーツ,家電,映画など）のWEBメディアごとにフォルダに記事がテキストファイルで保存されている
#     
# 以下、ファイルを読み込んで、必要な部分を抽出

# In[7]:


tsv_fname = "all_text.tsv" 
'''
#urllib.request.urlretrieve("https://www.rondhuit.com/download/ldcc-20140209.tar.gz", "ldcc-20140209.tar.gz")
# ダウンロードした圧縮ファイルのパスを設定
#tgz_fname = "ldcc-20140209.tar.gz" 
# 2つをニュースメディアのジャンルを選定
mydata = '/export/livedoor' 
#処理をした結果を保存するファイル名 
'''
def remove_brackets(inp):
    output = re.sub(u'[〃-〿]', '',(re.sub('＝|=|×|\(|\)|“|”|（|）|／|\[|\]| |　|…|・|\n|\t|/|＜|＞|@|＠', '', re.sub(u'[ℊ-⿻]', '', inp)))) #210A ~ 2FFF
    return output

def read_title(f):
    next(f)
    next(f)
    title = next(f)
    title = remove_brackets(title.encode().decode('utf-8'))
    return title[:-1]

def read_para(f):
    p = ''
    while True:
        try:
            para = next(f)
            para = remove_brackets(para.encode().decode('utf-8'))
            p += para
        except StopIteration:
            break
    return p [:-1]


# In[8]:


if articletype == 0:
    directory = ['/export/livedoor/dokujo-tsushin', '/export/livedoor/it-life-hack']
    target_genre = ["dokujo-tsushin", "it-life-hack"]
elif articletype == 1:
    directory = ['/export/livedoor/dokujo-tsushin', '/export/livedoor/peachy']
    target_genre = ["dokujo-tsushin", "peachy"]
else:
    print('No articles')
    exit()

zero_fnames = []
one_fnames = []

if os.path.exists(tsv_fname) == True:
    with open(tsv_fname, "r+") as f:
        f.truncate(0)

for i in range(2):
    for filename in os.listdir(directory[i]):
        if "LICENSE.txt" in filename:
            continue
        f = os.path.join(directory[i], filename)
#        if os.path.isfile(f):
#            print(f)
        if target_genre[0] in f and f.endswith(".txt"):
            with open(tsv_fname, "a") as wf:
                writer = csv.writer(wf, delimiter='\t')
                with open(f) as zf:
                    title = read_title(zf)
                    para = read_para(zf)
                    row = [target_genre[0], '0', title, para]
                    writer.writerow(row)
            continue
        if target_genre[1] in f and f.endswith(".txt"):
            with open(tsv_fname, "a") as wf:
                writer = csv.writer(wf, delimiter='\t')
                with open(f) as zf:
                    title = read_title(zf)
                    para = read_para(zf)
                    row = [target_genre[1], '1', title, para]
                    writer.writerow(row)
            continue


# pandasでデータを読み込み

# In[9]:


import pandas as pd
# データの読み込み
df = pd.read_csv("all_text.tsv", 
                 delimiter='\t', header=None, names=['media_name', 'label','title','sentence'])

# データの確認
#print(f'データサイズ： {df.shape}')
#df.sample(10)


# //文章データをsentences、ラベルデータを labelsに保存、以降この2変数だけを利用

# In[10]:


mn = df.media_name.values
labels = df.label.values
titles = df.title.values
sentences = df.sentence.values


# In[11]:


tagger = MeCab.Tagger("-Owakati")

def make_wakati(sentence):
  # MeCabで分かち書きを行う
    sentence = tagger.parse(sentence)
  # 半角全角英数字などは削除する
#    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
  # 記号なども削除する
#    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
  # スペース区切で形態素の配列に変換する
    wakati = sentence.split(" ")
  # 空要素を削除する
    wakati = list(filter(("").__ne__, wakati))
    return wakati


# In[12]:


wakati_sentences = []

for i in range(len(sentences)):
    wakati_sentences.append(make_wakati(sentences[i]))


# In[13]:


wcount = 256

emptylist = []
ssentences = np.append(emptylist, copy.deepcopy(sentences))

emplist = []
sectionlist = []

for i in enumerate(wakati_sentences):
    emp = 0
    section = 1
    if len(i[1])>wcount:
        wcount = 128
        count = 0
        countend = 0
        ssentences[i[0]] = []
        while len(i[1])-count-wcount>0:
            oneph = ''
            countend_ = 1
            while countend_%wcount != 0:
                oneph += i[1][countend]
                countend+=1
                countend_+=1
            ssentences[i[0]].append(oneph)
            count += wcount-1
            section += 1
        oneph = ''
        for j in range(len(i[1][count:-1])):
            oneph += i[1][count]
            count += 1
            emp += 1
        emplist.append(emp)
        ssentences[i[0]].append('')
        ssentences[i[0]][-1] = oneph
        sectionlist.append(section)
    else:
        oneph = ''
        for k in range(len(i[1])):
            oneph += i[1][k]
            ssentences[i[0]] = oneph
        emp = wcount - len(i[1])
        emplist.append(emp)
        sectionlist.append(1)


# In[14]:


#print(ssentences[0])


# # BERT Tokenizerを用いて単語分割・IDへ変換
# ## Tokenizerの準備
# 単語分割とIDへ変換

# # テスト実行

# In[15]:


w_input_ids = []
w_attention_masks = []

for sent in ssentences:
    p_input_ids = []
    p_attention_masks = []
    for sect in sent:
        sencoded_dict = tokenizer.encode_plus(
                            sect,                      
                            add_special_tokens = True, # Special Tokenの追加
                            max_length = wcount+2,  # I think maximum 文章の長さを固定（Padding/Trancatinating）
                            truncation=True,                
                            pad_to_max_length = True,# PADDINGで埋める
                            return_attention_mask = True,   # Attention maksの作成
                            return_tensors = 'pt',     #  Pytorch tensorsで返す
                       )
        p_input_ids.append(torch.tensor(sencoded_dict['input_ids']).view(-1))
        p_attention_masks.append(torch.tensor(sencoded_dict['attention_mask']).view(-1))
    w_input_ids.append(p_input_ids)
    w_attention_masks.append(p_attention_masks)


# In[16]:


# nagasa soroeru yo - id
pad = torch.full((1,130),3).view(-1)
maxlen = max(sectionlist)

for i in range(len(w_input_ids)):
    if maxlen>len(w_input_ids[i]):
        while maxlen>len(w_input_ids[i]):
            w_input_ids[i].append(pad)


# In[17]:


# nagasa soroeru yo - attention
pad = torch.full((1,130),0).view(-1)

for i in range(len(w_attention_masks)):
    if maxlen>len(w_attention_masks[i]):
        while maxlen>len(w_attention_masks[i]):
            w_attention_masks[i].append(pad)


# In[18]:


#len(w_input_ids)
#len(sectionlist)


# In[19]:


# 80%地点のIDを取得
num_dataset = len(w_input_ids)
train_size = numof_learn
val_size = num_dataset - train_size
#print('訓練データ数:{}'.format(train_size))
#print('検証データ数:{}'.format(val_size))


# In[20]:


from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split, RandomSampler, SequentialSampler
import torchvision.transforms as transforms
from transformers import RobertaForMaskedLM
import random

# データローダーの作成
transformmethods = []
if 'r' in transformflags:
    transformmethods.append(synreplace(synreplace_rate))
#    print("synreplace")
if 'i' in transformflags:
    transformmethods.append(randinsert(randinsert_rate))
#    print("randinsert")
if 'd' in transformflags:
    transformmethods.append(randdelete(randdelete_rate))
#    print("randdelete")
if 's' in transformflags:
    transformmethods.append(randswap(randswap_rate))
#    print("randswap")
data_transform = transforms.Compose(transformmethods)

class MyDatasets(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels, valids, transform=None):
        self.ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.valids = valids
        self.transform = transform
        
    def __getitem__(self, idx):
        xa, mask, label, valid = self.ids[idx], self.attention_mask[idx], self.labels[idx], self.valids[idx]
        if self.transform:
            xa = self.transform(xa)
        return xa, mask, [label]*len(xa), valid

    def __len__(self):
        return len(self.ids)

indices = np.random.choice(num_dataset, num_dataset, replace=False)

# データセットクラスの作成 ichigyoume ha randamu shitei
wt_input_ids = [w_input_ids[i] for i in indices[:train_size]]
wt_attention_masks = [w_attention_masks[i] for i in indices[:train_size]]
wt_labels = [labels[i] for i in indices[:train_size]]
wt_values = [sectionlist[i] for i in indices[:train_size]]
wv_input_ids = [w_input_ids[i] for i in indices[train_size:]]
wv_attention_masks = [w_attention_masks[i] for i in indices[train_size:]]
wv_labels = [labels[i] for i in indices[train_size:]]
wv_values = [sectionlist[i] for i in indices[train_size:]]

train_dataset = MyDatasets(wt_input_ids, wt_attention_masks, wt_labels, wt_values)
val_dataset = MyDatasets(wv_input_ids, wv_attention_masks, wv_labels, wv_values)

# データローダーの作成

# 訓練データローダー
# shuffle True/False to compare or not
train_dataloader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True
        )

# 検証データローダー
validation_dataloader = DataLoader(
            val_dataset, 
            batch_size = 1,
            shuffle = False
        )


# In[21]:


#a = wt_input_ids[0][0].detach().numpy()
# 50 kyoushi de-ta, saidai 46 block, 128+2
#a[0]


# In[22]:


from transformers import BertForSequenceClassification,AdamW,BertConfig

# BertForSequenceClassification 学習済みモデルのロード
model = BertForSequenceClassification.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking", # 日本語Pre trainedモデルの指定
    num_labels = 2, # ラベル数（今回はBinaryなので2、数値を増やせばマルチラベルも対応可）
    output_attentions = False, # アテンションベクトルを出力するか
    output_hidden_states = False, # 隠れ層を出力するか
).to(device)


# In[23]:


# 最適化手法の設定
optimizer = AdamW(model.parameters(), lr=2e-5)


# In[24]:


#test = next(iter(train_dataloader))


# In[25]:


# 学習の実行
train_loss_ = []
test_loss_ = []


# In[34]:


from tqdm import tqdm
from typing import OrderedDict

import scipy.stats as stats
def train(epoch, model):
    model.train() # 訓練モードで実行
    train_loss = 0
    with tqdm(train_dataloader) as pbar:
        pbar.set_description(f'[Epoch {epoch + 1}/{max_epoch}]')
        for ids, mask, labels, values in train_dataloader:# train_dataloaderはword_id, mask, labelを出力する点に注意
#        b_input_ids = ids.to(device)
#        b_input_mask = mask.to(device)
#        b_labels = labels.to(device)
            b_input_ids = torch.stack(ids).to(device)
            b_input_mask = torch.stack(mask).to(device)
            b_labels = torch.stack(labels).to(device)
            optimizer.zero_grad()
            alloutputs = []
            for n in values:
                outputs = []
                for i in range(n):
                    m_input_ids = b_input_ids[i]
                    m_input_mask = b_input_mask[i]
                    m_label = b_labels[i]
                    output = model(m_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=m_input_mask, 
                                 labels=m_label)
                    loss = output.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    pbar.set_postfix(
                        OrderedDict(
                            Loss=loss.item(),
                        )
                    )                    
                    train_loss += loss.item()
#                    print(output)
#                    print(output['logits'])
                    outputs.append(output['logits'].to('cpu'))
                alloutputs.append(outputs)
    return train_loss, alloutputs

def validation(model):
    model.eval()# 訓練モードをオフ
    val_loss = 0
    alloutputs = []
    with tqdm(train_dataloader) as pbar:
        with torch.no_grad(): # 勾配を計算しない
            for iteration, (ids, mask, labels, values) in enumerate(validation_dataloader):
        #        b_input_mask = mask.to(device)
        #        b_labels = labels.to(device)
                b_input_ids = torch.stack(ids).to(device)
                b_input_mask = torch.stack(mask).to(device)
                b_labels = torch.stack(labels).to(device)
                for n in values:
                    outputs = []
                    for i in range(n):
                        m_input_ids = b_input_ids[i]
                        m_input_mask = b_input_mask[i]
                        m_label = b_labels[i]
                        output = model(m_input_ids, 
                                     token_type_ids=None, 
                                     attention_mask=m_input_mask, 
                                     labels=m_label)
                        loss = output.loss
                        preds = output.logits.argmax(axis=1)
                        pbar.set_postfix(
                            OrderedDict(
                                Loss=loss.item(),
                                Accuracy=torch.sum(preds == b_labels).item() / len(b_labels),
                            )
                        )
                        outputs.append(output.logits.to('cpu').clone())
                    alloutputs.append(outputs)
                if numof_validation < iteration:
                    break
    return loss, alloutputs


# In[27]:


# nagasa soroeru yo
pad = torch.full((1,130),3).view(-1)
maxlen = max(sectionlist)

for i in range(len(w_input_ids)):
    if maxlen>len(w_input_ids[i]):
        while maxlen>len(w_input_ids[i]):
            w_input_ids[i].append(pad)


# In[28]:


for epoch in range(max_epoch):
    train_ = train(epoch, model)
    train_loss_.append(train_)
#    if epoch%10 == 0:
#        print('epoch: ', epoch)


# In[35]:


test_loss_ = validation(model)
# print('test: ', test_loss_)


# In[36]:


# b_input_mask.size(), b_input_ids.size(), labels.size()
# outputs = self.model(torch.unsqueeze(token_tensor, 0))


# In[37]:


test_loss_[0]# all loss
test_loss_[1] # 1690
test_loss_[1][0] # burokkusuu
test_loss_[1][0][0] # batch ikko niha shita
test_loss_[1][0][0][0] # hoshii yatsu


# In[38]:


test_loss_[1][0][4][0]


# In[39]:


len(wv_labels)


# # HOUHOU 1

# In[40]:


methodone = []
for i in range(len(test_loss_[1])):
    article = []
    for j in range(len(test_loss_[1][i])):
        block = np.argmax(test_loss_[1][i][j][0].numpy())
        article.append(block)
    articlesum = np.sum(np.array(article))
    if articlesum/len(test_loss_[1][i]) <= 0.5:
        methodone.append(0)
    else:
        methodone.append(1)


# # HOUHOU2

# In[41]:


methodtwo = []
for i in range(len(test_loss_[1])):
    article = [0,0]
    for j in range(len(test_loss_[1][i])):
        block = test_loss_[1][i][j][0].numpy()
        article = [x+y for (x,y) in zip(article, block)]
    articlesum = np.argmax(np.array(article))
    if articlesum <= 0.5:
        methodtwo.append(0)
    else:
        methodtwo.append(1)


# In[42]:


# nanka houhou 2 ga umaku ittenai kamo
# seikai tono hikaku shitai ne
len(methodtwo)


# In[43]:


one_df = pd.DataFrame(methodone, columns=['method1'])
two_df = pd.DataFrame(methodtwo, columns=['method2'])
label_df = pd.DataFrame(wv_labels, columns=['true_label'])
accuracy_df = pd.concat([one_df, two_df, label_df], axis=1)
accuracy_df.head(50)


# In[44]:


from sklearn.metrics import f1_score
def accuracy(pdf):
    return (pdf == label_df.values[:len(pdf)]).sum()/len(pdf)

def fscore(pdf):
    return f1_score(pdf, label_df.values[:len(pdf)])


# In[51]:


import csv
f = open('ens_augens-re2-rep-'+filestr+'.csv', 'w')
onepreds = one_df.values
twopreds = two_df.values
f.write('methodone: acc:'+str(accuracy(onepreds))+', f1:'+str(fscore(onepreds))+'\n')
f.write('methodtwo: acc:'+str(accuracy(twopreds))+', f1:'+str(fscore(twopreds))+'\n')
f.close()
