#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torchvision import models, transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import cv2
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import ViTFeatureExtractor, ViTModel
from transformers import BertModel
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
from transformers import AdamW
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import  mean_absolute_error
import os, sys
import glob
sys.path.append('early-stopping-pytorch')
from pytorchtools import EarlyStopping
from torch.nn.utils.rnn import pad_packed_sequence
import math
# % matplotlib inline





torch.cuda.empty_cache()





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)





# Default dataset files
data_dir = '<path to images>'
train_path = '<path to trainset>'
dev_path = '<path to devset>'
test_path = '<path to testset>'





train_df = pd.read_json(train_path, lines=True)
val_df = pd.read_json(dev_path, lines=True)
test_df = pd.read_json(test_path, lines=True)





# Creation of word2indxx dictionary for learning target embeddings from scratch 

test_path1 = '/home/shivams/multimodal/Res/Manual_Test/3set_new/test_all.jsonl'
test_df1 = pd.read_json(test_path1, lines=True)
test_path2 = '/home/shivams/multimodal/Res/Manual_Test/3set_new/test_unseenall.jsonl'
test_df2 = pd.read_json(test_path2, lines=True)
test_path3 = '/home/shivams/multimodal/Res/Manual_Test/3set_new/test_unseenharmful.jsonl'
test_df3 = pd.read_json(test_path3, lines=True)
all_df = pd.concat([train_df, val_df, test_df1, test_df2, test_df3])

# Get all possible target set
all_tar_list = []
for item in all_df['text']:
#     print(item[1])
    all_tar_list.append(item[1].strip().lower())
#     break

# len(all_tar_list)
all_tar_list_copy = all_tar_list
all_tar_set = list(set(all_tar_list_copy))
all_tar_set.sort()
all_tar_set[:10]


V = len(all_tar_set)

word_to_ix = dict()
for ix, item in enumerate(all_tar_set):
    word_to_ix[item] = ix





# For CUSTOMISED OCR, target
ocrs_train = []
targets_train = []
labels_train = []
for row in train_df.iterrows():
    ocr = row[1]['text'][0]
    tar = row[1]['text'][1]
    label_text = row[1]['labels'][0]
    if label_text=='not harmful':
        label = 0
    else:
        label = 1
    ocrs_train.append(ocr)
    targets_train.append(tar)

    labels_train.append(label)

ocrs_validation = []
targets_validation = []
labels_validation = []
for row in val_df.iterrows():
    ocr = row[1]['text'][0]
    tar = row[1]['text'][1]
    label_text = row[1]['labels'][0]
    if label_text=='not harmful':
        label = 0
    else:
        label = 1
    ocrs_validation.append(ocr)
    targets_validation.append(tar)
    labels_validation.append(label)
ocrs_test = []
targets_test = []
labels_test = []
con1_test = []
con2_test = []
for row in test_df.iterrows():
    ocr = row[1]['text'][0]
    tar = row[1]['text'][1]
    label_text = row[1]['labels'][0]
    if label_text=='not harmful':
        label = 0
    else:
        label = 1
    ocrs_test.append(ocr)
    targets_test.append(tar)
    labels_test.append(label)


# ## CLIP model




#@title

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "../../bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    
clip_model = torch.jit.load("../../clip_model.pt").cuda().eval()
input_resolution = clip_model.input_resolution.item()
context_length = clip_model.context_length.item()
vocab_size = clip_model.vocab_size.item()

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
    ])
clip_tokenizer = SimpleTokenizer()

# Get the image features for a single image input
def process_image_clip(in_img):
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    
    image = preprocess(Image.open(in_img).convert("RGB"))
    
    image_input = torch.tensor(np.stack(image)).cuda()
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    return image_input
# Get the text features for a single text input
def process_text_clip(in_text):    
    text_token = clip_tokenizer.encode(in_text)
    text_input = torch.zeros(clip_model.context_length, dtype=torch.long)
    sot_token = clip_tokenizer.encoder['<|startoftext|>']
    eot_token = clip_tokenizer.encoder['<|endoftext|>']
    tokens = [sot_token] + text_token[:75] + [eot_token]
    text_input[:len(tokens)] = torch.tensor(tokens)
    text_input = text_input.cuda()
    return text_input





from sentence_transformers import SentenceTransformer
model_sent_trans = SentenceTransformer('paraphrase-distilroberta-base-v1')


# # Dataset class




# ocr, target using bert
from transformers import BertTokenizer, BertForSequenceClassification
try:
    del tokenizer
except:
    pass
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenised_train = tokenizer(ocrs_train, targets_train, max_length=256, padding=True, truncation=True)
tokenised_validation = tokenizer(ocrs_validation, targets_validation, max_length=256, padding=True, truncation=True)
tokenised_test = tokenizer(ocrs_test, targets_test, max_length=256, padding=True, truncation=True)

train_inputs = tokenised_train['input_ids']
validation_inputs = tokenised_validation['input_ids']
test_inputs = tokenised_test['input_ids']

train_labels = labels_train
validation_labels = labels_validation
test_labels = labels_test

train_masks = tokenised_train['attention_mask']
validation_masks = tokenised_validation['attention_mask']
test_masks = tokenised_test['attention_mask']

train_type_ids = tokenised_train['token_type_ids']
validation_type_ids = tokenised_validation['token_type_ids']
test_type_ids = tokenised_test['token_type_ids']

# Convert all of our data into torch tensors, the required datatype for our model

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
test_inputs = torch.tensor(test_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
test_labels = torch.tensor(test_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
test_masks = torch.tensor(test_masks)
train_type_id = torch.tensor(train_type_ids)
validation_type_id = torch.tensor(validation_type_ids)
test_type_id = torch.tensor(test_type_ids)





class HarmemeMemesDatasetAug(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path,
        img_dir,
        mode=None,
        balance=False,
        dev_limit=None,
        random_state=0,
    ):

        self.samples_frame = pd.read_json(
            data_path, lines=True
        )
        
        if mode == 'train':
            self.data_input = train_inputs
#             self.data_labels = train_labels
            self.data_masks = train_masks
            self.data_type_id = train_type_id
        elif mode == 'val':
            self.data_input = validation_inputs
#             self.data_labels = validation_labels
            self.data_masks = validation_masks
            self.data_type_id = validation_type_id
        else:
            self.data_input = test_inputs
#             self.data_labels = test_labels
            self.data_masks = test_masks
            self.data_type_id = test_type_id
        

        self.samples_frame = self.samples_frame.reset_index(
            drop=True
        )
        self.samples_frame.image = self.samples_frame.apply(
            lambda row: (img_dir + '/' + row.image), axis=1
        )
#         self.samples = ['harmful', 'not harmful']
#         self.label_encoder = encoders.LabelEncoder(self.samples, reserved_labels=['unknown'], unknown_index=0)

        
            
#         self.image_transform = image_transform
#         self.text_transform = text_transform

    def __len__(self):
        """This method is called when you do len(instance) 
        for an instance of this class.
        """
        return len(self.samples_frame)

    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.samples_frame.loc[idx, "id"]
        

#         ***Process for CLIP image input***
        image_clip_input = process_image_clip(self.samples_frame.loc[idx, "image"])
        context_clip_input = process_text_clip(self.samples_frame.loc[idx, "context"][0])
#         ***Process for fresh target embedding***
        cur_tar = self.samples_frame.loc[idx, "text"][1].strip().lower()
        tar_lookup_tensor = torch.tensor(word_to_ix[cur_tar], dtype=torch.long).to(device)
        input_ids = self.data_input[idx]
        input_mask = self.data_masks[idx]
        type_id = self.data_type_id[idx]
        if "labels" in self.samples_frame.columns:
#             Uncoment below for binary index creation
            if self.samples_frame.loc[idx, "labels"][0]=="not harmful":
                lab=0
            else:
                lab=1            
            label = torch.tensor(lab).to(device)  
            sample = {
                "id": img_id, 
                "image_clip_input": image_clip_input,
                "b_input_ids": input_ids,
                "b_input_mask": input_mask,
                "b_type_id": type_id,
                "context_clip_input": context_clip_input,
                "tar_lookup_tensor": tar_lookup_tensor,
                "label": label
            }
        else:
            sample = {
                "id": img_id, 
                "image_clip_input": image_clip_input,
                "b_input_ids": input_ids,
                "b_input_mask": input_mask,
                "b_type_id": type_id,
                "context_clip_input": context_clip_input,
                "tar_lookup_tensor": tar_lookup_tensor
            }

        return sample





BS = 4 #at least 10 can be tried (12327MiB being used)

hm_dataset_train = HarmemeMemesDatasetAug(train_path, data_dir, mode = 'train')
dataloader_train = DataLoader(hm_dataset_train, batch_size=BS,
                        shuffle=True, num_workers=0)
hm_dataset_val = HarmemeMemesDatasetAug(dev_path, data_dir, mode = 'val')
dataloader_val = DataLoader(hm_dataset_val, batch_size=BS,
                        shuffle=True, num_workers=0)
hm_dataset_test = HarmemeMemesDatasetAug(test_path, data_dir, mode = 'test')
dataloader_test = DataLoader(hm_dataset_test, batch_size=BS,
                        shuffle=False, num_workers=0)





try:
    del model
except:
    pass
# -------------------------------------------------------------------------------
# # MODEL definition
class MM(nn.Module):
    def __init__(self, n_out):
        super(MM, self).__init__()
        self.input_dim1 = 256
        self.embed_dim1 = 512
        self.model_out1 = 512
        
        self.input_dim2 = 256
        self.embed_dim2 = 512
        self.model_out2 = 512
        
        self.dropout = 0.10

        self.embeds = nn.Embedding(V, 300)
        self.text_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.relu = nn.ReLU()
        self.feat11_lrproj = nn.Linear(512, 256)
        self.feat12_lrproj = nn.Linear(512, 256)
        self.feat21_lrproj = nn.Linear(512, 256)
        self.feat22_lrproj = nn.Linear(512, 256)
        self.tanh = torch.nn.Tanh()
        # Concat layer for the combined feature space 
        self.dense512 = nn.Linear(300, 512) 
        self.dense1 = nn.Linear(1280, 512)
#         self.dense_tar = nn.Linear(768, 512)
#         self.dropout10 = nn.Dropout(0.10)              
        self.dense2 = nn.Linear(512, 256)         
        self.output_fc = nn.Linear(256, n_out)   
        self._create_outputnets()
    
    
    def _create_outputnets(self):
        self.output_net1 = nn.Sequential(
            nn.Linear(self.input_dim1, self.embed_dim1),
            nn.LayerNorm(self.embed_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim1, self.model_out1)
        )
        self.output_net2 = nn.Sequential(
            nn.Linear(self.input_dim2, self.embed_dim2),
            nn.LayerNorm(self.embed_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim2, self.model_out2)
        )
    
    
    def MLB1(self, feat1, feat2):
        feat1_lr = self.feat11_lrproj(feat1)
        feat2_lr = self.feat12_lrproj(feat2)
        z = torch.mul(feat1_lr, feat2_lr)
        z_out = self.tanh(z)
        mm_feat = self.output_net1(z_out)
        return mm_feat
    
    def MLB2(self, feat1, feat2):
        feat1_lr = self.feat21_lrproj(feat1)
        feat2_lr = self.feat22_lrproj(feat2)
        z = torch.mul(feat1_lr, feat2_lr)
        z_out = self.tanh(z)
        mm_feat = self.output_net2(z_out)
        return mm_feat

    def forward(self, clip_img_in, tar_lookup_in, clip_context_in, input_ids, type_id, input_mask):
        tar_512_rep = F.relu(self.dense512(self.embeds(tar_lookup_in)))
        context_tar_feat = self.MLB1(tar_512_rep, clip_context_in)
        
        bert_outputs = self.text_model(input_ids, token_type_ids=type_id, attention_mask=input_mask)
        bert_pooled_out = bert_outputs['pooler_output']

        feat_vect1 = torch.cat((bert_pooled_out, context_tar_feat), 1)
        out_dense1 = F.relu(self.dense1(feat_vect1))
        mm_feat_out = self.MLB2(out_dense1, clip_img_in)
        out_dense2 = F.relu(self.dense2(mm_feat_out))
        out = torch.sigmoid(self.output_fc(out_dense2))        
        return out 
# -------------------------------------------------------------------------------

output_size = 1 #Binary case
model = MM(output_size)
model.to(device)
print(model)
exp_name = "Exp_DISARM"
exp_path = "<path_to_your_project>/"+exp_name
lr=0.0001
criterion = nn.BCELoss() #Binary case
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)





# from prettytable import PrettyTable
# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad: continue
#         param = parameter.numel()
#         table.add_row([name, param])
#         total_params+=param
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params
    
# count_parameters(model)





# continued frmo before
from pathlib import Path
# For BCE loss
def train_model(model, patience, n_epochs):
    epochs = n_epochs
#     clip = 5

    train_acc_list=[]
    val_acc_list=[]
    train_loss_list=[]
    val_loss_list=[]
    
    # initialize the experiment path
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # initialize early_stopping object
    chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)

    model.train()
    for i in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for data in dataloader_train:
            b_input_ids = data["b_input_ids"].to(device)
            b_input_mask = data["b_input_mask"].to(device)
            b_type_id = data["b_type_id"].to(device).long()
            
# #             Clip features...
            img_inp_clip = data['image_clip_input']
            context_inp_clip = data['context_clip_input']
            with torch.no_grad():
                img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                context_feat_clip = clip_model.encode_text(context_inp_clip).float().to(device)

            tar_lookup_tensor = data['tar_lookup_tensor'].to(device)
            label = data['label'].to(device)
            model.zero_grad()
            output = model(img_feat_clip, tar_lookup_tensor, context_feat_clip, b_input_ids, b_type_id, b_input_mask)
            loss = criterion(output.squeeze(), label.float())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = torch.abs(output.squeeze() - label.float()).view(-1)
                acc = (1. - acc.sum() / acc.size()[0])
                total_acc_train += acc
                total_loss_train += loss.item()

        train_acc = total_acc_train/len(dataloader_train)
        train_loss = total_loss_train/len(dataloader_train)
        model.eval()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for data in dataloader_val:
                b_input_ids = data["b_input_ids"].to(device)
                b_input_mask = data["b_input_mask"].to(device)
                b_type_id = data["b_type_id"].to(device).long()
#                 Clip features...                
                img_inp_clip = data['image_clip_input']
                context_inp_clip = data['context_clip_input']
                with torch.no_grad():
                    img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                    context_feat_clip = clip_model.encode_text(context_inp_clip).float().to(device)
                tar_lookup_tensor = data['tar_lookup_tensor'].to(device)
                label = data['label'].to(device)
                model.zero_grad()
                output = model(img_feat_clip, tar_lookup_tensor, context_feat_clip, b_input_ids, b_type_id, b_input_mask)
                val_loss = criterion(output.squeeze(), label.float())
                acc = torch.abs(output.squeeze() - label.float()).view(-1)
                acc = (1. - acc.sum() / acc.size()[0])
                total_acc_val += acc
                total_loss_val += val_loss.item()
        print("Saving model...")         
        torch.save(model.state_dict(), os.path.join(exp_path, "final.pt"))
        val_acc = total_acc_val/len(dataloader_val)
        val_loss = total_loss_val/len(dataloader_val)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'Epoch {i+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        model.train()
        torch.cuda.empty_cache()
        
    # load the last checkpoint with the best model
#     model.load_state_dict(torch.load(chk_file))
    
    return  model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, i
        
n_epochs = 25

# early stopping patience; how long to wait after last time validation loss improved.
patience = 25

model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, epoc_num = train_model(model, patience, n_epochs)





# For BCE loss
def test_model(model):
    model.eval()
    total_acc_test = 0
    total_loss_test = 0
    outputs = []
    test_labels=[]
    with torch.no_grad():
        for data in dataloader_test:
            b_input_ids = data["b_input_ids"].to(device)
            b_input_mask = data["b_input_mask"].to(device)
            b_type_id = data["b_type_id"].to(device)
            img_inp_clip = data['image_clip_input']
            context_inp_clip = data['context_clip_input']
            with torch.no_grad():
                img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                context_feat_clip = clip_model.encode_text(context_inp_clip).float().to(device)
            
            tar_lookup_tensor = data['tar_lookup_tensor'].to(device)
            label = data['label'].to(device)
            out= model(img_feat_clip, tar_lookup_tensor, context_feat_clip, b_input_ids, b_type_id, b_input_mask)
            outputs += list(out.cpu().data.numpy())
            if out.to(device).shape[0]>1:
                out_data = out.squeeze()
            else:
                out_data = out[0]
            loss = criterion(out_data, label.float())
            acc = torch.abs(out_data - label.float()).view(-1)
            acc = (1. - acc.sum() / acc.size()[0])
            total_acc_test += acc
            total_loss_test += loss.item()

    acc_test = total_acc_test/len(dataloader_test)
    loss_test = total_loss_test/len(dataloader_test)
    print(f'acc: {acc_test:.4f} loss: {loss_test:.4f}')
    return outputs





# Loead from a checkpoint
# try:
#     del model
# except:
#     pass
# # path = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
# path = os.path.join(exp_path, "final.pt")
# model = MM(output_size)
# model.load_state_dict(torch.load(path))
# model.to(device)





outputs = test_model(model)





# All in a go
np_out = np.array(outputs)
y_pred = np.zeros(np_out.shape)
y_pred[np_out>0.0005]=1
# y_pred[np_out>0.2]=1
y_pred = np.array(y_pred)

# try:
#     test_labels=labels_test
# except:
test_labels = []
for index, row in test_df.iterrows():
    lab = row['labels'][0]
    if lab=="not harmful":
        test_labels.append(0)
    else:
        test_labels.append(1)
        
rec = np.round(recall_score(test_labels, y_pred, average="macro"),4)
prec = np.round(precision_score(test_labels, y_pred, average="macro"),4)
f1 = np.round(f1_score(test_labels, y_pred, average="macro"),4)
acc = np.round(accuracy_score(test_labels, y_pred),4)
roc_auc = np.round(roc_auc_score(test_labels, y_pred), 4)
print("Acc, Prec, Rec, F1, roc-auc")
print(acc, prec, rec, f1, roc_auc)
# print(tabulate([[acc, prec, rec, f1, roc_auc]], headers=["Acc", "Prec", "Rec", "F1", "roc-auc"]))
print()
print(classification_report(test_labels, y_pred))

