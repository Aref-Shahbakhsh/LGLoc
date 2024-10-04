import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from Bio import SeqIO
import numpy as np
import random
dna = []
label = []



def con(s):
  return '/content/drive/MyDrive/tez/dataset/'+s
index = 0
paths = ['Cytoplasm_train.fasta','Endoplasmic_reticulum_train.fasta','Extracellular_region_train.fasta' ,
         'Mitochondria_train.fasta','Nucleus_train.fasta']
# paths = list(map(con,paths))
classes = [0,1,2,3,4]
for path in paths :
    sequence = SeqIO.parse(path,'fasta')

    for record in sequence:
        A = str(record.seq)
        dna.append(A)
        label.append( classes[index])

    index = index + 1




dna2 = []
label2 = []
index2 = 0

paths2 = ['Cytoplasm_indep.fasta','Endoplasmic_reticulum_indep.fasta' , 'Extracellular_region_indep.fasta'
,'Mitochondria_indep.fasta','Nucleus_indep.fasta']

# paths2 = list(map(con,paths2))



for path in paths2:
    sequence = SeqIO.parse(path,'fasta')


    for record in sequence:
        A = str(record.seq)
        dna2.append(A)
        label2.append( classes[index2])


    index2 = index2 + 1


print(dna[0])

dna_new = []
label_new = []
dna2_new = []
label2_new = []

for i in range (len(dna)):
  if len(dna[i])<=80000:
    dna_new.append(dna[i])
    label_new.append(label[i])
for i in range (len(dna2)):
  if len(dna2[i])<=80000:
    dna2_new.append(dna2[i])
    label2_new.append(label2[i])

dna = dna_new
label = label_new
dna2 = dna2_new
label2 = label2_new

def select(inp):
  if len(inp)<=1022:
    return inp
  else:
    return inp[0:511] + inp[-511:]
dna = list(map(select , dna))
dna2 = list(map(select , dna2))





k = 6
def sliding(s):
  temp = ''
  for i in range(0, len(s), k):
    temp  = temp + s[i:i+k] + ' '
  return temp

"""// tavali ro bayad be soorat kmer dar biyaram ya na ?"""

import pandas as pd
from datasets import Dataset
train_file = pd.DataFrame({'dna':dna,'label':label})
test_file = pd.DataFrame({'dna':dna2,'label':label2})
train_dataset = Dataset.from_pandas(train_file)
test_dataset = Dataset.from_pandas(test_file)

#from transformers import AutoTokenizer, AutoModel , AutoModelForSequenceClassification
#tokenizer = AutoTokenizer.from_pretrained('dnagpt/human_gpt2-v1')
#model = AutoModelForSequenceClassification.from_pretrained('dnagpt/human_gpt2-v1' , num_labels=5)
#print(model)

# gpn
from transformers import AutoTokenizer, AutoModelForSequenceClassification , AutoModel
import gpn.model

model_path = "songlab/gpn-brassicales"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path , num_labels=5)









def preprocess_function(examples , segment_length = 3060 , stride = 300):
    data_set = []
    label_data_set = []
    for example in examples:
      dna_sequence = example['dna']
      label = example['label']
      for start in range(0, len(dna_sequence), stride):
         end = min(start + segment_length, len(dna_sequence))
         segment = dna_sequence[start:end]
        #  segment = convert_T_to_U(segment)
        # segment = ' '.join(segment)
         segment = sliding(segment)
        #  encoded_inputs = tokenizer(segment, padding='max_length', return_tensors="pt")
        #  encoded_inputs['labels'] = torch.tensor(label)
         data_set.append(segment)
         label_data_set.append(label)
    return pd.DataFrame({'dna':data_set,'label':label_data_set})
#train_tokenized_datasets   = preprocess_function(train_dataset)
#test_tokenized_datasets  = preprocess_function(test_dataset)




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
model = model.to('cuda')

def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
       if hasattr(layer, 'reset_parameters'):
           print(f'Reset trainable parameters of layer = {layer}')
           layer.reset_parameters()

#model.apply(reset_weights)





criterion = FocalLoss(alpha=1.5 , gamma=2)  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Define optimizer.
from torch.utils.data import DataLoader
data_loader = DataLoader(list(zip(dna ,label)) , batch_size = 28 , shuffle = True)
epoch = 400
best_acc = 0.0
best_f1 = 0.0
best_per_f1  = 0.0
def save_best_model(model, per_f1):
    print(f"Saving model with best_acc: {best_acc}")
    torch.save({
                'model_state_dict': model.state_dict()
                }, 'outputs/CNN_GPN.pth')
for i in range(epoch):
  model.train()
  epoch_loss = 0.0
  j = 0
  for  i , data in enumerate(tqdm(data_loader)):
    x = data[0]
    y = data[1].to('cuda')
    del data
    optimizer.zero_grad()
    x = tokenizer(x, return_tensors='pt' , truncation=True , max_length=1024 , padding=True).to('cuda')
    x = x["input_ids"]


    output = model(x).logits
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
    test_loader = DataLoader(list(zip(dna2 , label2)) , batch_size = 28 , shuffle = False)
    for i , data  in enumerate(tqdm(test_loader)):
        x = data[0]
        y = data[1].to('cuda')

        x = tokenizer(x, return_tensors='pt' , truncation=True , max_length=1024 , padding=True).to('cuda')
        x = x["input_ids"]

        logits = model(x).logits
        del x
        logits = logits.softmax(dim=-1)
        logits = logits.argmax(dim=1)
        pred.append(logits)
  pred = torch.cat(pred , dim=0)
  print(classification_report(pred.to('cpu') , label2))
  accuracy = classification_report(pred.to('cpu') , label2, output_dict=True)['accuracy']

  f1 = classification_report(pred.to('cpu') , label2, output_dict=True)['weighted avg']['f1-score']
  per = classification_report(pred.to('cpu') , label2, output_dict=True)['1']['f1-score']
  if accuracy > best_acc:
        save_best_model(model, per)
        best_acc = accuracy
        best_f1 = f1
  print('Epoch:', epoch,'best_accuracy:', best_acc , 'f1:' , best_f1 , 'best_per:' , best_per_f1  ,  'Loss:', epoch_loss / len(dna))

