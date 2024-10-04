
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch_sparse
from torch_sparse import SparseTensor
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import torch
import forgi.graph.bulge_graph as fgb
import torch.nn as nn
import numpy as np
from lion_pytorch import Lion

base_label = {'A':0 , 'C':1 , 'G':2 , 'T':3 }
# base_color_lookup = {'A':'r' , 'C':'b' , 'G':'y' , 'U':'g' }
base_color_lookup = {'A':0 , 'C':1 , 'G':2 , 'T':3 }  # Modified to map colors to integers
base_lookup = {'A':[1,0,0,0] , 'C':[0,1,0,0] , 'G':[0,0,1,0] , 'T':[0,0,0,1] }
struct_lookup = {'f':[1,0,0,0,0,0] , 't':[0,1,0,0,0,0],'s':[0,0,1,0,0,0] ,'i':[0,0,0,1,0,0] , 'm':[0,0,0,0,1,0] , 'h':[0,0,0,0,0,1] }
# base_lookup = {'A':[135.13 ,1,1,1,0.103,3.8,-110.53 , 73.76] , 'C':[111.10,1,2,1,1.49,4.5,-130.18,91.67] , 'G':[151.13 ,2,1,1,0.051,9.2,-135.41,83.83] , 'U':[112.04,1,2,1,2.07,9.3,-222.8,68.01] 
              # , 'T': [126.11 , 1,2,1,0.223,9.7,-132.83,67.36]}
edge_lookup = {'phosphodiester_bond' : 1 , 'base_pairing': 2 , 'single':0 }

import time
def construct_rna_graph(sequence, dotbracket , device='cpu'):
    bg= fgb.BulgeGraph.from_dotbracket(dotbracket)
    struct = bg.to_element_string()
    G = nx.Graph()
    # Add nodes
    for i, base in enumerate(sequence):
        G.add_node(i, x=base_lookup[base] + struct_lookup[struct[i]]  , color = base_color_lookup[sequence[i]] )


    # Add edges from phosphodiester bonds
    if len(sequence) - 1 != 0:
      for i in range(len(sequence) - 1):
          G.add_edge(i, i + 1, bond_type=edge_lookup['phosphodiester_bond'])


    # Add edges from base pairing bonds
    stack = []
    for i, bracket in enumerate(dotbracket):
        if bracket == '(':
            stack.append(i)  # Push the index to the stack
        elif bracket == ')':
            if stack:
                paired_index = stack.pop()  # Get the paired index
                G.add_edge(i, paired_index, bond_type=edge_lookup['base_pairing'])


    # Convert to PyTorch Geometric Data
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    # Add reverse edges
    reverse_edge_index = torch.flip(edge_index, [0])
    edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)


    # if nx.is_connected(G):
    #   diameter = nx.diameter(G)
    #   print('Diameter:', diameter)
    # else:
    #   print('Graph is not connected.')




    # Convert node attributes to Numpy arrays
    x_np = np.array([G.nodes[i]['x'] for i in G.nodes])
    color_np = np.array([G.nodes[i]['color'] for i in G.nodes])

    # Convert edge attributes to Numpy arrays
    bond_type_np = np.array([G.edges[i, j]['bond_type'] for i, j in G.edges])

    # Convert Numpy arrays to Torch tensors
    x = torch.from_numpy(x_np)
    color = torch.from_numpy(color_np)
    bond_type = torch.from_numpy(bond_type_np)

    # Add bond types for reverse edges
    bond_type = torch.cat((bond_type, bond_type))

    data = Data(x=x, edge_index=edge_index, color=color, bond_type=bond_type).to(device)
    del G, x_np, color_np, bond_type_np, reverse_edge_index 

    return   data



from collections import Counter
def get_kmers_and_frequency(sequence, k):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return kmers , Counter(kmers)



def construct_rna_graph2(sequence , device='cpu' ,  k=9):
# Create an empty graph
  G = nx.Graph()
  kmers , counts = get_kmers_and_frequency(sequence, k)
  # Add nodes (k-mers) to the graph

  for kmer in range(len(kmers)):
    k_mer_one_hot = [bit for base in kmers[kmer] for bit in base_lookup[base]]
    if len(k_mer_one_hot) < k*4:
      k_mer_one_hot += [0] * (k*4 - len(k_mer_one_hot))
    G.add_node(kmer , x = [counts[kmers[kmer]]] + k_mer_one_hot )


  # Add edges to the graph
  for i in range(len(kmers) - 1):
      if kmers[i][1:] == kmers[i+1][:-1]:
          G.add_edge(i, i+1)
  x_np = np.array([G.nodes[i]['x'] for i in G.nodes])
  x = torch.from_numpy(x_np)


  # Convert to PyTorch Geometric Data
  edge_index = torch.tensor(list(G.edges)).t().contiguous()
    # Add reverse edges
  reverse_edge_index = torch.flip(edge_index, [0])
  edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
  data = Data(x=x, edge_index=edge_index).to(device)

  del G, x_np , x , edge_index , reverse_edge_index

  return   data








def convert_T_to_U(rna):
  sequence = ''.join([c.replace('T','U') for c in rna])
  return sequence


# arraye haye dna  label ,dna2  label2 ,  baraye train set va test set  halat 1/6 , 5/6 ee .
# arraye haye dnafull , labelfull , tamame sequense haye mrna ee .
from Bio import SeqIO
#import pandas as pd
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
  if len(dna[i])<20000:
    dna_new.append(dna[i])
    label_new.append(label[i])
for i in range (len(dna2)):
  if len(dna2[i])<20000:
    dna2_new.append(dna2[i])
    label2_new.append(label2[i])
dna = dna_new
label = label_new
dna2 = dna2_new
label2 = label2_new
label = list(map(lambda x : 1 if x==4 else 0  , label))
label2 = list(map(lambda x : 1 if x==4 else 0  , label2))

import pandas as pd
from datasets import Dataset
train_file = pd.DataFrame({'dna':dna,'label':label})
test_file = pd.DataFrame({'dna':dna2,'label':label2})
train_dataset = Dataset.from_pandas(train_file)
test_dataset = Dataset.from_pandas(test_file)


# from transformers import BertForSequenceClassification , BigBirdTokenizer , BertTokenizer ,  AutoTokenizer  , AutoModelForSequenceClassification , AutoModel , BigBirdForSequenceClassification

import torch
# num_classes = 5
# tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNA_bert_6' , trust_remote_code=True)
# model = BertForSequenceClassification.from_pretrained('zhihan1996/DNA_bert_6',num_labels=5)
# tokenizer.add_tokens(all_kmers)
# tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
# model = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base' , num_labels=num_classes)
# SPLICEBERT_PATH = "/content/SpliceBERT/models/SpliceBERT.1024nt"
# tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_PATH)
# bert_model = AutoModel.from_pretrained(SPLICEBERT_PATH) # assume the class number is 3
# model.resize_token_embeddings(len(tokenizer))

# Define a function to preprocess the data


def preprocess_function(examples , segment_length = 1024 , stride = 300):
    data_set = []
    label_data_set = []
    for example in examples:
      dna_sequence = example['dna']
      label = example['label']
      # for start in range(0, len(dna_sequence), stride):
        # end = min(start + segment_length, len(dna_sequence))
        # segment = dna_sequence[start:end]
        # segment = convert_T_to_U(segment)
        # segment = ' '.join(segment)
        # segment = sliding(segment)
        # encoded_inputs = tokenizer(segment, padding='max_length', return_tensors="pt")
        # encoded_inputs['labels'] = torch.tensor(label)
        # data_set.append(encoded_inputs)
      data_set.append(dna_sequence)
      label_data_set.append(label)
    return pd.DataFrame({'dna':data_set,'label':label_data_set})





train_tokenized_datasets   = preprocess_function(train_dataset)
test_tokenized_datasets  = preprocess_function(test_dataset)


from torch_geometric.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_list, transform=None, pre_transform=None):
        super(MyDataset, self).__init__('.', transform, pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv , conv
from torch_geometric.nn import global_max_pool , global_add_pool , global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout , LSTM , Tanh
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn.models import GraphUNet



import torch
from torch.nn import Linear, BatchNorm1d, Dropout, ReLU
from torch_geometric.nn import GCNConv, GATConv , SAGEConv
from torch_geometric.nn import aggr , pool , conv , dense
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_scatter import scatter_mean, scatter_max , scatter_add

def readout(x, batch):
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0)

    return torch.cat((x_mean, x_max), dim=-1)
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # First sequence
        self.conv1 = conv.GENConv(in_channels=10 ,out_channels=100)  # increase output dimension
        self.conv2 = conv.GENConv(in_channels=100,out_channels=100)
        self.conv3 = conv.GENConv(in_channels=100,out_channels=100)







        self.dropout = Dropout(p=0.2)
        self.fc1 = Linear(200,100)

        self.classifier = Linear(100,2)

        self.relu = ReLU()

    def forward(self, x, edge_index ,  batch):
        # First sequence

        x = self.relu(self.conv1(x, edge_index))
        #x, edge_index, edge_weight, batch, perm = self.pool1(x=x, edge_index=edge_index, edge_weight=None, batch=batch)
        x1 = readout(x,batch)

        x = self.relu(self.conv2(x=x, edge_index=edge_index))
        #x, edge_index, edge_weight, batch, perm = self.pool2(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        x2 = readout(x,batch)

        x = self.relu(self.conv3(x=x, edge_index=edge_index))
        #x, edge_index, edge_weight, batch, perm = self.pool3(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        x3 = readout(x,batch)



        encode = self.relu(x1) + self.relu(x2) + self.relu(x3)
        #encode = torch.cat((self.relu(x1) , self.relu(x2) , self.relu(x3) ), dim=1)

        x = self.relu(self.fc1(encode))
        x = self.dropout(x)
        x = self.classifier(x)

        return x , encode




import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool





class GIN(torch.nn.Module):
    def __init__(self, dim_h):
        super(GIN, self).__init__()
#        self.powermean_aggr1 = aggr.PowerMeanAggregation(learn=True)
 #       self.powermean_aggr2 = aggr.PowerMeanAggregation(learn=True)
  #      self.powermean_aggr3 = aggr.PowerMeanAggregation(learn=True)


        self.conv1 = GINConv(
            Sequential(Linear(10, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*2, dim_h)
        self.lin2 = Linear(dim_h,2)
        self.relu = ReLU()


    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = readout(h1, batch)
        h2 = readout(h2, batch)
        h3 = readout(h3, batch)

        # Concatenate graph embeddings
        #encode = torch.cat((h1, h2, h3), dim=1)
        encode = self.relu(h1) + self.relu(h2)+ self.relu(h3)

        # Classifier
        h = self.lin1(encode)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h,encode











model = GCN()
print(model)

import gzip
import pickle
fold_test = np.load('folded_test_set_primary.npy')

from tqdm import tqdm
test_graphs = []
from tqdm import tqdm
for i in tqdm(range(len(test_tokenized_datasets.dna))):
  test_graphs.append(construct_rna_graph(test_tokenized_datasets.dna[i] , fold_test[i][0],device='cuda'))
print('end')

import gzip
import pickle
with gzip.open('train_graphs_primary.pkl.gz', 'rb') as f:
    train_graphs = pickle.load(f)



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
        



import time
import multiprocessing as mp
from torch_geometric.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
model = GCN().to('cuda')
criterion = FocalLoss(alpha=1.5 , gamma=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Define optimizer.

from concurrent.futures import ProcessPoolExecutor
def train(train_graphs , epoch_loss):
    model.train()
    for i , data in enumerate(train_graphs):
      data , labels = data[0] , data[1]


      batch_index = data.batch
      optimizer.zero_grad()  # Clear gradients.
      labels = labels.to('cuda')
      x = data.x.float().to('cuda')
      edge = data.edge_index.to('cuda')
      index = batch_index.to('cuda')
      out , _ = model(x, edge , index)


      loss = criterion(out.view(-1,2), labels)  # Compute the loss solely based on the training nodes.
      epoch_loss =epoch_loss+ loss.item()  # Accumulate the loss for each iteration

      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      del x ,edge , index , out , labels

      return epoch_loss




best_acc = 0.0
best_f1 = 0.0

def save_best_model(model, accuracy):
    print(f"Saving model with accuracy: {accuracy}")
    torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'f1':f1 ,
                }, 'outputs/GNN.pth')



def eval(test_graphs):
  pred = []
  model.eval()
  with torch.no_grad():
    for data , labels in test_graphs:


      batch_index = data.batch.to('cuda')
      labels = labels.to('cuda')
      x = data.x.float().to('cuda')
      edge = data.edge_index.to('cuda')
      index = batch_index.to('cuda')

      out , _ = model(x, edge ,  index)


      logits = out.view(-1,2)
      logits = logits.softmax(-1)
      logits = logits.argmax(dim=1)
      pred.append(logits)
      predication = torch.cat(pred , dim=0).to('cpu')
  print(classification_report(predication , test_tokenized_datasets.label))
  accuracy = classification_report(predication , test_tokenized_datasets.label, output_dict=True)['accuracy']
  f1 = classification_report(predication , test_tokenized_datasets.label, output_dict=True)['weighted avg']['f1-score']

  return accuracy , f1







train_set = MyDataset(train_graphs)
test_set = MyDataset(test_graphs)

train_set = DataLoader(list(zip(train_set , label)) , batch_size=64 , shuffle=True)
test_set = DataLoader(list(zip(test_set , label2)) , batch_size=64 , shuffle=False)

for epoch in range(900000):
  epoch_loss = 0.0
  epoch_loss = train(train_set , epoch_loss)
  if epoch%50==0:
     accuracy , f1  = eval(test_set)
     if accuracy > best_acc:
        save_best_model(model, accuracy)
        best_acc = accuracy
        best_f1 = f1
  print('Epoch:', epoch,'best_acc:', best_acc , 'f1:' , best_f1 ,   'Loss:', epoch_loss / len(dna))
