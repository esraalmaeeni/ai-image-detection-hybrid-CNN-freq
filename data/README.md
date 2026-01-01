## Dataset

This project uses a fixed, stratified subset derived from the
"AI vs Human Generated Dataset" (Kaggle).

### Source
https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset

### Subset Construction
- Stratified by label
- Train/Eval ratio: 80/20
- Random seed: 42
- Total samples: 15,990

The subset was created using `scripts/create_subset.py`.

### Reproducibility
To reproduce the dataset:
1. Download the full dataset from Kaggle
2. Run `scripts/create_subset.py`
3. Place the resulting `data/` directory at project root

No manual selection was performed.
