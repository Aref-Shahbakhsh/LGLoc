
from Bio import SeqIO
import pandas as pd
import numpy as np
import random
import pandas as pd

dna = []
label = []

def con(s):
  return '/content/drive/MyDrive/tez/dataset/'+s

index = 0

paths = ['train/Cytoplasm_train.fasta','train/Endoplasmic_reticulum_train.fasta' ,
         'train/Extracellular_region_train.fasta','train/Mitochondria_train.fasta' , 'train/Nucleus_train.fasta']
paths = list(map(con,paths))
classes = [0,1,2,3,4]
# temp = []

for path in paths :

    sequence = SeqIO.parse(path,'fasta')
    # a = pd.DataFrame(Weighted_kmer(path , k=2 , pk=2).Generate_weighted_kmer_Features())
    # b = pd.DataFrame(Weighted_kmer(path , k=3 , pk=3).Generate_weighted_kmer_Features())
    # c = pd.DataFrame(Weighted_kmer(path , k=4 , pk=4).Generate_weighted_kmer_Features())
    # d = pd.DataFrame(Weighted_kmer(path , k=5 , pk=5).Generate_weighted_kmer_Features())
    # e = pd.DataFrame(Weighted_kmer(path , k=6 , pk=6).Generate_weighted_kmer_Features())







    # temp.append(pd.concat([a,b,c,d , e] ,axis=1))


    for record in sequence:
        A = str(record.seq)
        dna.append(A)
        label.append( classes[index])
    index = index + 1


# temp[1:] = list(map(lambda x : x.drop( 0 , axis=0) , temp[1:]))
# dna_kmer = pd.concat(temp , axis=0)
# label = [-1] + label
# dna_kmer['label'] = label


# temp = []

dna2 = []
label2 = []
index2 = 0
# dna_kmer2 = []

paths2 = ['Cytoplasm_indep.fasta','Endoplasmic_reticulum_indep.fasta' ,
         'Extracellular_region_indep.fasta','Mitochondria_indep.fasta' , 'Nucleus_indep.fasta']


paths2 = list(map(con,paths2))

for path in paths2 :

    sequence = SeqIO.parse(path,'fasta')
    # a = pd.DataFrame(Weighted_kmer(path , k=2 , pk=2).Generate_weighted_kmer_Features())
    # b = pd.DataFrame(Weighted_kmer(path , k=3 , pk=3).Generate_weighted_kmer_Features())
    # c = pd.DataFrame(Weighted_kmer(path , k=4 , pk=4).Generate_weighted_kmer_Features())
    # d = pd.DataFrame(Weighted_kmer(path , k=5 , pk=5).Generate_weighted_kmer_Features())
    # e = pd.DataFrame(Weighted_kmer(path , k=6 , pk=6).Generate_weighted_kmer_Features())

    # temp.append(pd.concat([a,b,c,d ,e] ,axis=1))


    for record in sequence:
        A = str(record.seq)
        dna2.append(A)
        label2.append( classes[index2])
    index2 = index2 + 1


# temp[1:] = list(map(lambda x : x.drop( 0 , axis=0) , temp[1:]))
# dna_kmer2 = pd.concat(temp , axis=0)
# label2 = [-1] + label2

# dna_kmer2['label'] = label2

# dna_kmer = dna_kmer.drop(0 , axis=1)
# dna_kmer2 = dna_kmer2.drop(0 , axis=1)
# dna_kmer = dna_kmer.T.reset_index(drop=True).T
# dna_kmer2 = dna_kmer2.T.reset_index(drop=True).T
# coulmns_name = dna_kmer.loc[0]
# dna_kmer = dna_kmer.drop(0,axis=0)
# dna_kmer2 = dna_kmer2.drop(0,axis=0)
# del coulmns_name[5456]
# del label[-1]
# del label2[-1]
# # dna_kmer = dna_kmer.drop(1360 , axis=1)
# # dna_kmer2 = dna_kmer2.drop(1360 , axis=1)
# dna_kmer = dna_kmer.drop(5456 , axis=1)
# dna_kmer2 = dna_kmer2.drop(5456 , axis=1)
# print(dna[0])



def select(inp):
  if len(inp)<=1022:
    return inp
  else:
    return inp[0:511] + inp[-511:]
dna = list(map(select , dna))
dna2 = list(map(select , dna2))
def splice(seq):
  return ' '.join(list(seq.upper().replace("U", "T")))

dna = list(map(splice , dna))
dna2 = list(map(splice , dna2))
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification , AutoModel

SPLICEBERT_PATH = "/content/drive/MyDrive/tez/SpliceBERT.1024nt"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(SPLICEBERT_PATH, num_labels=5 , output_attentions=True )
# model = AutoModel.from_pretrained(SPLICEBERT_PATH)
#Weights = '/content/drive/MyDrive/tez/splice_bert.pth'
#model.load_state_dict(torch.load(Weights , map_location=torch.device('cpu')))
# Train Splice bert
import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)


from sklearn.metrics import classification_report
from tqdm import tqdm
# model = model.to('cuda')
criterion = FocalLoss(alpha=1.5 , gamma=2)  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # Define optimizer.
# havaset be lr basheee
from torch.utils.data import DataLoader
data_loader = DataLoader(list(zip(dna , label)) , batch_size = 12 , shuffle = True)
epoch = 6
for i in range(epoch):
  model.train()
  epoch_loss = 0.0
  j = 0
  for  i , data in enumerate(tqdm(data_loader)):
    x = data[0]
    y = data[1].to('cuda')
    del data
    optimizer.zero_grad()
    x = tokenizer(x, return_tensors='pt' , truncation=True , padding=True , max_length=1024).to('cuda')
    # x = x["input_ids"]
    output = model(**x).logits
    del x

    loss = criterion(output.view(-1,5) , y)
    del output , y
    epoch_loss += loss
    loss.backward()
    optimizer.step()
    j = j+1
  print('epoch =' , i , ' loss = ' , epoch_loss/j)

  with torch.no_grad():
    pred = []
    model.eval()
    test_loader = DataLoader(list(zip(dna2 , label2)) , batch_size = 12 , shuffle = False)
    for i , data  in enumerate(tqdm(test_loader)):
        x = data[0]
        y = data[1].to('cuda')

        x = tokenizer(x, return_tensors='pt' , truncation=True  , padding=True ,max_length=1024).to('cuda')
        # x = x["input_ids"]
        logits = model(**x).logits
        del x
        logits = logits.softmax(dim=-1)
        logits = logits.argmax(dim=1)
        pred.append(logits)
  pred = torch.cat(pred , dim=0)
  print(classification_report(pred.to('cpu') , label2))
  from sklearn.metrics import confusion_matrix
  confusion = confusion_matrix(label2, pred.to('cpu'))

  TN = confusion[0, 0]
  FP = confusion[0, 1]
  FN = confusion[1, 0]
  TP = confusion[1, 1]
  sensitivity = TP / (TP + FN)
  specificity = TN / (TN + FP)
  accuracy = (TP + TN) / (TP + TN + FP + FN)
  print(f"Class {0} - Sensitivity: {sensitivity}, Specificity: {specificity}, Accuracy: {accuracy}")




#pred splice
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
model = model.to('cuda')
with torch.no_grad():
  pred_splice = []
  model.eval()
  test_loader = DataLoader(list(zip(dna2 , label2)) , batch_size = 64 , shuffle = False)
  for i , data  in enumerate(tqdm(test_loader)):
      x = data[0]
      y = data[1].to('cuda')

      x = tokenizer(x, return_tensors='pt' , truncation=True  , padding=True ,max_length=1024).to('cuda')
      # x = x["input_ids"]
      logits = model(**x).logits
      del x
      logits = logits.softmax(dim=-1)
      # logits = logits.argmax(dim=1)
      pred_splice.append(logits)
pred_splice = torch.cat(pred_splice , dim=0)
# print(classification_report(pred_splice.to('cpu') , label2))


#Embedding splice test
embedding_splice_dna2 = []
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
model = model.to('cuda')
with torch.no_grad():
  pred_splice = []
  model.eval()
  loader = DataLoader(list(zip(dna2 , label2)) , batch_size = 1 , shuffle = False)
  for i , data  in enumerate(tqdm(loader)):
      x = data[0]
      y = data[1].to('cuda')

      x = tokenizer(x, return_tensors='pt' , truncation=True  , padding=True ,max_length=1024).to('cuda')
      # x = x["input_ids"]
      embedding = model(**x).hidden_states[-1]
      embedding_mean = torch.mean(embedding[0], dim=0)
      # embedding_max = torch.max(embedding[0], dim=0)[0]
      # emb = torch.cat((embedding_mean , embedding_max))
      embedding_splice_dna2.append(embedding_mean.to('cpu'))


#Embedding splice train
embedding_splice_dna = []
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
model = model.to('cuda')
with torch.no_grad():
  pred_splice = []
  model.eval()
  loader = DataLoader(list(zip(dna , label)) , batch_size = 1 , shuffle = False)
  for i , data  in enumerate(tqdm(loader)):
      x = data[0]
      y = data[1].to('cuda')

      x = tokenizer(x, return_tensors='pt' , truncation=True  , padding=True ,max_length=1024).to('cuda')
      # x = x["input_ids"]
      embedding = model(**x).hidden_states[-1]
      embedding_mean = torch.mean(embedding[0], dim=0)
      # embedding_max = torch.max(embedding[0], dim=0)[0]
      # emb = torch.cat((embedding_mean , embedding_max))
      embedding_splice_dna.append(embedding_mean.to('cpu'))
