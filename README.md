# LGLoc

LGLoc is a bioinformatics tool designed for mRNA localization prediction using a combination of Graph Neural Networks (GNN), BERT model, and secondary structure analysis.

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

The `Model` folder contains the original implementation of:

- Graph Neural Network (GNN)
- BERT (Bidirectional Encoder Representations from Transformers)

### 🔧 Usage

You can train these models on the provided LGLoc dataset or your own dataset.

### 📥 Pretrained Weights

Pretrained weights are available in the `Model_weights` directory. You can load these to avoid retraining from scratch.

---

## 🔬 Secondary Structure Prediction

The secondary structures used in LGLoc are stored in the `Secondary_structure` folder and were predicted using the [ViennaRNA Package](https://www.tbi.univie.ac.at/RNA/).

To predict secondary structure for your own RNA sequences:

```python
import RNA

seq = 'AAACCCGGGTTT'  # Replace with your sequence
structure, energy = RNA.fold(seq)
print(structure)
```

---

## 📊 Feature Vectors

- GNN and BERT features are stored in the `feature vectore from gnn and bert` folder.
- These features are generated using trained GNN and BERT models.

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

> [Paper Title Here]  
> [Authors Here]  
> [Link or DOI]

---

## 📁 Project Structure

```
LGLoc/
│
├── Model/                 # GNN and BERT model definitions
├── Model_weights/        # Pretrained model weights
├── Secondary_structure/  # Predicted RNA secondary structures
├── feature vectores from bert and gnn/         # GNN and BERT feature vectors
├── performance_evaluation.py
└── requirements.txt
```

---

## 📫 Contact

For questions, suggestions, or collaboration, feel free to open an issue or contact us via [email/aref.shahbakhsh1998@gmail.com]
