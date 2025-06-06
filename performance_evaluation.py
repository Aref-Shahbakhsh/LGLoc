# -*- coding: utf-8 -*-

import re
from collections import Counter
import itertools
import pandas as pd
import numpy as np

'''
Python class to generate imporved Kmer features based on formulas in PLEK papper which uses
improved Kmer scheme to representDNA seqences in Fasta file. The link to teh paper is
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4177586/pdf/12859_2013_Article_6586.pdf
'''
class Weighted_kmer:
    def __init__(self, fasta_file, k, pk, normalize=True):
        self.file = fasta_file
        self.k = k
        self.pk = pk
        self.normalize = normalize
        self.nucleotides='ACGT'

    def readfasta(self):
        with open(self.file) as f:
            records = f.read()
        if re.search('>', records) == None:
            print('Error,the input DNA sequence must be fasta format.')
            sys.exit(1)
        records = records.split('>')[1:]
        myFasta = []
        for fasta in records:
            array = fasta.split('\n')
            name, sequence = array[0].split()[0], re.sub('[^ACGT-]', '-', ''.join(array[1:]).upper())
            myFasta.append([name, sequence])
        return myFasta

    def generate_list(self):
        ACGT_list=["".join(e) for e in itertools.product(self.nucleotides, repeat=self.k)]
        return ACGT_list

    def kmerArray(self,sequence):
        kmer = []
        for i in range(len(sequence) - self.k + 1):
            kmer.append(sequence[i:i + self.k])
        return kmer

    #def Generate_PLEK_Kmer_Features(self,file, k=2, PK=3, normalize=True):
    def Generate_weighted_kmer_Features(self):
        '''
            Generate imporved weighted Kmer features
            input_data: input DNA Fasta file : Text file with seq id and seq of ACTG
            K : Kmer value (1,2,3,4,5,6,) : integer value
            normalize: boolean if TRUE normalize: COUNT / Seq-length
        '''
        fastas=self.readfasta()
        w = 1.0 / (4**(self.pk - self.k))
        print("w = ", w)
        vector = []
        header = ['id']
        if self.k < 1:
            print('k must be an integer and greater than 0.')
            return 0
        for kmer in itertools.product(self.nucleotides, repeat=self.k):
            header.append(''.join(kmer))
        vector.append(header)
        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            kmers = self.kmerArray(sequence)
            count = Counter()
            count.update(kmers)
            if self.normalize == True:
                for key in count:
                    count[key] = (count[key] / len(kmers)) * w
            code = [name]
            for j in range(1, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            vector.append(code)
        return vector

#In the first part we Generate K-mer Features
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
temp = []

for path in paths :

    sequence = SeqIO.parse(path,'fasta')
    a = pd.DataFrame(Weighted_kmer(path , k=2 , pk=2).Generate_weighted_kmer_Features())
    b = pd.DataFrame(Weighted_kmer(path , k=3 , pk=3).Generate_weighted_kmer_Features())
    c = pd.DataFrame(Weighted_kmer(path , k=4 , pk=4).Generate_weighted_kmer_Features())
    d = pd.DataFrame(Weighted_kmer(path , k=5 , pk=5).Generate_weighted_kmer_Features())
    e = pd.DataFrame(Weighted_kmer(path , k=6 , pk=6).Generate_weighted_kmer_Features())



    temp.append(pd.concat([a,b,c,d ,e] ,axis=1))


    for record in sequence:
        A = str(record.seq)
        dna.append(A)
        label.append( classes[index])
    index = index + 1


temp[1:] = list(map(lambda x : x.drop( 0 , axis=0) , temp[1:]))
dna_kmer = pd.concat(temp , axis=0)
label = [-1] + label
dna_kmer['label'] = label


temp = []

dna2 = []
label2 = []
index2 = 0
dna_kmer2 = []

paths2 = ['Cytoplasm_indep.fasta','Endoplasmic_reticulum_indep.fasta' ,
         'Extracellular_region_indep.fasta','Mitochondria_indep.fasta' , 'Nucleus_indep.fasta']


paths2 = list(map(con,paths2))

for path in paths2 :

    sequence = SeqIO.parse(path,'fasta')
    a = pd.DataFrame(Weighted_kmer(path , k=2 , pk=2).Generate_weighted_kmer_Features())
    b = pd.DataFrame(Weighted_kmer(path , k=3 , pk=3).Generate_weighted_kmer_Features())
    c = pd.DataFrame(Weighted_kmer(path , k=4 , pk=4).Generate_weighted_kmer_Features())
    d = pd.DataFrame(Weighted_kmer(path , k=5 , pk=5).Generate_weighted_kmer_Features())
    e = pd.DataFrame(Weighted_kmer(path , k=6 , pk=6).Generate_weighted_kmer_Features())


    temp.append(pd.concat([a,b,c,d,e] ,axis=1))


    for record in sequence:
        A = str(record.seq)
        dna2.append(A)
        label2.append( classes[index2])
    index2 = index2 + 1

temp[1:] = list(map(lambda x : x.drop( 0 , axis=0) , temp[1:]))
dna_kmer2 = pd.concat(temp , axis=0)
label2 = [-1] + label2

dna_kmer2['label'] = label2

dna_kmer = dna_kmer.drop(0 , axis=1)
dna_kmer2 = dna_kmer2.drop(0 , axis=1)
dna_kmer = dna_kmer.T.reset_index(drop=True).T
dna_kmer2 = dna_kmer2.T.reset_index(drop=True).T
coulmns_name = dna_kmer.loc[0]
dna_kmer = dna_kmer.drop(0,axis=0)
dna_kmer2 = dna_kmer2.drop(0,axis=0)
del coulmns_name[5456]
del label[0]
del label2[0]
dna_kmer = dna_kmer.drop(5456 , axis=1)
dna_kmer2 = dna_kmer2.drop(5456 , axis=1)

import itertools

# Define the four DNA bases
bases = ['A', 'C', 'G', 'T']

# Define the k-mer sizes
k_sizes = [2, 3, 4, 5 , 6]

# Initialize the list to store all k-mer feature names
feature_names = []

# For each k-mer size
for k in k_sizes:
    # Generate all possible k-mers and add them to the feature names list
    feature_names.extend([''.join(p) for p in itertools.product(bases, repeat=k)])


import pandas as pd
import numpy as np

# Load your dataset into a pandas DataFrame
dataset = dna_kmer
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset.columns = feature_names
# Calculate the correlation matrix
correlation_matrix = dataset.corr()

# Identify the highly correlated features
threshold = 0.8
highly_correlated_features = np.where(np.abs(correlation_matrix) > threshold)

# Remove the highly correlated features
features_to_drop = set()
for i, j in zip(*highly_correlated_features):
    if i != j and i not in features_to_drop and j not in features_to_drop:
        features_to_drop.add(j)

dataset_without_highly_correlated_features = dataset.drop(dataset.columns[list(features_to_drop)], axis=1)
without_dna_kmer  = dataset_without_highly_correlated_features

dataset = dna_kmer2
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset.columns = feature_names
without_dna_kmer2 = dataset.drop(dataset.columns[list(features_to_drop)], axis=1)

snap_train = ['/content/drive/MyDrive/tez/SNAP/snap_CP_train.csv',
               '/content/drive/MyDrive/tez/SNAP/snap_ER_train.csv',
                '/content/drive/MyDrive/tez/SNAP/snap_EXR_train.csv',
                '/content/drive/MyDrive/tez/SNAP/snap_MC_train.csv',
                '/content/drive/MyDrive/tez/SNAP/snap_NC_train.csv'
                ]
snap_test = ['/content/drive/MyDrive/tez/SNAP/snap_CP_indep.csv',
               '/content/drive/MyDrive/tez/SNAP/snap_ER_indep.csv',
                '/content/drive/MyDrive/tez/SNAP/snap_EXR_indep.csv',
                '/content/drive/MyDrive/tez/SNAP/snap_MC_indep.csv',
                '/content/drive/MyDrive/tez/SNAP/snap_NC_indep.csv'
                ]
snap_dna = []
for i in snap_train:
  snap_dna = snap_dna + list((pd.read_csv(i , header=None).drop(0 , axis=1).to_numpy()))

snap_dna2 = []
for i in snap_test:
  snap_dna2 = snap_dna2 + list((pd.read_csv(i , header=None).drop(0 , axis=1).to_numpy()))

# Load Features that come from Splicebert model

np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
embedding_splice_dna = np.load('/content/drive/MyDrive/tez/emb_splice_dna.npy')
embedding_splice_dna2 = np.load('/content/drive/MyDrive/tez/emb_splice_dna2.npy')
np.load = np_load_old

# Load features that come from GNN

# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# mslpdna = np.load('/content/mslpdna.npy')
# mslpdna2 = np.load('/content/mslpdna2.npy')
dna_gnn_emb = np.load('/content/drive/MyDrive/tez/dna_gnn_emb.npy')
dna2_gnn_emb = np.load('/content/drive/MyDrive/tez/dna2_gnn_emb.npy')

np.load = np_load_old
dna_gnn_emb = list(map(lambda x : x[0] , dna_gnn_emb))
dna2_gnn_emb = list(map(lambda x : x[0] , dna2_gnn_emb))

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from sklearn import metrics

# Evaluation Metrics Functions
def compute_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    mcc = (tp*tn - fp*fn) / (np.sqrt(  (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)  ) + 1e-8)
    return round(mcc,3)


def compute_sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp/(tp+fn)
    return round(sensitivity,3)

def compute_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn/(fp+tn)
    return round(specificity,3)

def compute_accuracy(y_true, y_pred):
    accuracy = (y_true==y_pred).sum()/len(y_true)
    return round(accuracy,3)

def compute_precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tn/(tn+fp)
    return round(precision,3)

def compute_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="macro" , zero_division=1)
    return round(f1,3)
def compute_auc(y_train , y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    return round(auc,3)





dna_new = []
label_new = []
dna2_new = []
label2_new = []
old_label = label
old_label2 = label2
list_dna_number_orginal = []
list_dna_number_no_orginal = []
for i in range (len(dna)):
  list_dna_number_orginal.append(i)

  if len(dna[i])<=20000:
    dna_new.append(dna[i])
    label_new.append(label[i])
    list_dna_number_no_orginal.append(i)

  else:
    print('train set:' , i )
    print('label is:',label[i])


for i in range (len(dna2)):
  if len(dna2[i])<=20000:
    dna2_new.append(dna2[i])
    label2_new.append(label2[i])
  else:
    print('test set:' , i)
    print('label is:',label[i])
dna = dna_new
label = label_new
dna2 = dna2_new
label2 = label2_new

# concate GNN with splice bert

dna_gnn_splice = []
dna_gnn_splice2 = []
for  a , b    in list(zip(dna_gnn_emb ,  embedding_splice_dna )):


  temp =  a.to('cpu').tolist() + b.to('cpu').tolist()
  dna_gnn_splice.append(temp)

for  a , b  in list(zip(dna2_gnn_emb , embedding_splice_dna2  )):


  temp =  a.to('cpu').tolist() + b.to('cpu').tolist()
  dna_gnn_splice2.append(temp)

  # concate K-mer feature with cksnap

kmer_cksnap = []
kmer_cksnap2 = []
for  a , b   in list(zip(without_dna_kmer.to_numpy() , snap_dna )):


  temp =  list(a) + list(b)
  kmer_cksnap.append(temp)

for  a , b  in list(zip(without_dna_kmer2.to_numpy() , snap_dna2 )):


  temp =  list(a) + list(b)
  kmer_cksnap2.append(temp)

copy_label = label
copy_label2 = label2

priors = [None , [0.3,0.7] , [0.2,0.8] , None , None]
K = [len(dna_gnn_splice[0]),len(dna_gnn_splice[0]),450 , 85 , 90]
smoth = [1e-9,1e-2, 1e-2,1e-9,1e-9]
Class_name = ['CP','ER','EXR','MC','NC']

# Now we evaluate every model based on their own features
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import  SelectKBest , f_classif
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
GLOBAL_RANDOM_STATE = 42

from sklearn.naive_bayes import MultinomialNB , BernoulliNB , CategoricalNB , ComplementNB , MultinomialNB , GaussianNB
for i in range(5):
  if i!=2:
    label = copy_label
    label2 = copy_label2
    label = list(map(lambda x : 1 if x==i else 0  , label))
    label2 = list(map(lambda x : 1 if x==i else 0  , label2))
    final_combine_embedding_dnat = dna_gnn_splice
    final_combine_embedding_dna2t = dna_gnn_splice2
  else:
    label = old_label
    label2 = old_label2
    label = list(map(lambda x : 1 if x==i else 0  , label))
    label2 = list(map(lambda x : 1 if x==i else 0  , label2))
    final_combine_embedding_dnat = kmer_cksnap
    final_combine_embedding_dna2t = kmer_cksnap2






  # Feature Selection

  X = final_combine_embedding_dnat
  y = label
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True , random_state=GLOBAL_RANDOM_STATE)
  model = SelectKBest(score_func = f_classif , k=K[i]).fit(X_train, y_train)
  clf_0 =  GaussianNB(priors= priors[i] , var_smoothing=smoth[i])
  clf_0.fit(model.transform(final_combine_embedding_dnat), label)
  if i==1:
    clf_svm = SVC(class_weight='balanced' , kernel='linear' , probability=True)
    clf_svm.fit(final_combine_embedding_dnat, label)
    predication_svm = clf_svm.predict_proba(final_combine_embedding_dna2t)
    predication_gaussian = clf_0.predict_proba(model.transform(final_combine_embedding_dna2t))
    comb_probability = [(p1 + p2) / 2 for p1, p2 in zip(predication_svm, predication_gaussian)]
    y_pred_proba = np.array(comb_probability)[:, 1]
    predication_naive_0 = np.argmax(comb_probability , axis=1)
  else:
      predication_naive_0 = clf_0.predict(model.transform(np.array(final_combine_embedding_dna2t)))
      y_pred_proba = clf_0.predict_proba(model.transform(np.array(final_combine_embedding_dna2t)))[:, 1]

  print('Class_name: '+Class_name[i])
  mcc = compute_mcc(label2, predication_naive_0)
  sen = compute_sensitivity(label2, predication_naive_0)
  spe = compute_specificity(label2, predication_naive_0)
  acc = compute_accuracy(label2, predication_naive_0)
  pre = compute_precision(label2, predication_naive_0)
  F1  = compute_f1(label2, predication_naive_0)
  print('Sensitivity : {}'.format(round(sen,3)))
  print('Specifity : {}'.format(round(spe,3)))
  print('accuracy : {}'.format(round(acc,3)))
  print('Precision : {}'.format(round(pre,3)))
  print('F1 : {}'.format(round(F1,3)))
  print('mcc : {}'.format(round(mcc,3)))
  print('auc:' , roc_auc_score(label2, y_pred_proba, multi_class="ovr"))
