# LGLoc

LGLoc is a bioinformatics tool designed for mRNA localization prediction using a combination of Graph Neural Networks (GNN), BERT model, and secondary structure analysis.
![alt text](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-03485-8/MediaObjects/41598_2025_3485_Fig1_HTML.png?as=webp)


---

## 📁 Project Structure

```
LGLoc/
│
├── Models/                             # GNN and BERT model definitions
├── Model weights/                      # Pretrained model weights
├── predicted secondary structures/     # Predicted RNA secondary structures
├── Feature vectors from GNN and BERT/  # GNN & BERT feature vectors
├── CKSNAP features/                    # CKSNAP feature vectors
├── K-mer Encoder/                      # K-mer encoding files
├── performance_evaluation.py
├── Raw Dataset.txt
├── requirements.txt
└── LICENSE

```

---

## 📦 Installation

Before running LGLoc, install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Performance Evaluation

To evaluate the performance of LGLoc on a dataset, use the following command:

```bash
python performance_evaluation.py
```

Make sure to set the correct dataset path inside the script or via configuration.

---

## 🧠 GNN and BERT Models

The `Models` folder contains the original implementation of:

- Graph Neural Network (GNN)
- BERT (Bidirectional Encoder Representations from Transformers)

### 🔧 Usage

You can train these models on the provided LGLoc dataset or your own dataset.

### 📥 Pretrained Weights

Pretrained weights are available in the `Model weights` directory. You can load these to avoid retraining from scratch.

---

## 🔬 Secondary Structure Prediction

The secondary structures used in LGLoc are stored in the `predicted secondary structures` folder and were predicted using the [ViennaRNA Package](https://www.tbi.univie.ac.at/RNA/).

To predict secondary structure for your own RNA sequences:

```python
import RNA

seq = 'AAACCCGGGUUU'  # Replace with your sequence
structure, energy = RNA.fold(seq)
print(structure)
```

---

## 📊 Feature Vectors
- CKSNAP features are stored in the CKSNAP features folder.
- K-mer encodings are available in the K-mer Encoder folder.
- GNN and BERT features are stored in the `Feature vectors from GNN and BERT` folder, these features are generated using trained GNN and BERT models.

### 🛠 Generate Your Own Features

To generate feature vectors on a custom dataset:

```bash
python GNN.py         # For GNN features
python Splice_Bert.py # For BERT features
```

Ensure you configure your dataset path correctly in each script.

---

## 📄 Citation

If you use LGLoc in your research or application, please cite our LGLoc paper. 
> [[Link to the paper](https://www.nature.com/articles/s41598-025-03485-8)]

---

## 📫 Contact

For questions, suggestions, or collaboration, feel free to open an issue or contact us via [aref.shahbakhsh1998@gmail.com].
