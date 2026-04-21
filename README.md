# Graph Neural Networks for Predicting Chromatin Activation in Craniofacial Development

**Can we predict which regulatory elements will activate during cranial neural crest cell (CNCC) migration using graph neural networks on chromatin interaction networks?**

This project applies GNNs to CHiCAGO-scored chromatin interaction data from [our Nature Communications paper] on Polycomb-mediated chromatin topology in craniofacial development.

## Key Finding

> **Epigenetic features — not chromatin topology — drive activation prediction.** Through systematic experiments varying graph architecture, edge modeling, and feature engineering, we found that the H3K27ac/H3K27me3 balance is the primary predictor of element activation. 3D chromatin interactions at E8.5 add minimal predictive power, suggesting the causal enhancer-promoter contacts form *during* the developmental transition, not before it.

## Experiments

| Experiment | Architecture | Features | AUROC | AUPRC | Key Insight |
|-----------|-------------|----------|-------|-------|-------------|
| Baseline | GAT (binary edges, th=5) | 3 ChIP marks | 0.751 | 0.382 | Initial model, high recall / low precision |
| Threshold tuning | GAT (th=3) | 3 ChIP marks | 0.751 | 0.383 | Denser graph ≠ better prediction |
| Connected only | GAT (th=5) | 3 ChIP marks | 0.744 | 0.096 | Topology alone is insufficient |
| Continuous edges | TransformerConv | 3 ChIP marks | 0.766 | 0.399 | Edge features help modestly |
| **Enriched features** | **TransformerConv** | **10 engineered** | **0.789** | **0.424** | **Feature engineering > architecture tuning** |

## Repository Structure

```
├── notebooks/
│   └── GNN_Chromatin_Activation_Analysis.ipynb   # Full analysis walkthrough
├── src/
│   ├── data_loader.py       # Graph construction with feature engineering
│   ├── model.py             # GNN architectures (GAT, TransformerConv, WeightedGCN)
│   ├── train.py             # Training loop with class-weighted loss
│   └── evaluate.py          # Metrics, UMAP, confusion matrices
├── configs/                 # YAML experiment configurations
├── figures/                 # Generated visualizations
├── scripts/
│   └── extract_coordinates.R  # Extract genomic coords from SE objects
└── requirements.txt
```

## Data

- **108,162 nodes**: 39,707 promoters + 68,455 enhancers
- **Edges**: CHiCAGO-scored chromatin interactions (PCHi-C)
- **Node features**: H3K27ac, H3K4me2, H3K27me3 ChIP-seq signals at E8.5
- **Task**: Binary classification — will a non-active element at E8.5 become active by E10.5?
- **Class imbalance**: ~19% positive (activated), ~81% negative

## Technical Stack

PyTorch, PyTorch Geometric, scikit-learn, UMAP, matplotlib, seaborn

## Author

Yousra Ben Zouari — [Nature Communications paper]
