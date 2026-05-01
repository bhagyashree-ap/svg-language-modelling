Scaling Laws for SVG Language Models: µP Optimized Transformer Training and Vector Graphic Generation

Goal:
Train transformer models to generate SVGs, evaluate generation quality, and analyze scaling and µP vs SP behavior.

High level flow:
- Prepare and clean raw SVGs locally (Part 1).
- Train small experiments and scaling runs on Colab (Parts 2 and 3).
- Train the best model, generate samples, compute metrics, and save artifacts on Colab (Part 4).

Workflow:
Part 1: .py file run locally to preprocess SVGs, train tokenizer, and produce tokenized datasets.  
Parts 2, 3, 4: .ipynb files run on Google Colab using the Drive folder produced by Part 1.

Folder Structure:

svg_project_root/<br/>
├── part1_local/<br/>
│   ├── part4-dataset.py<br/>
│   ├── part1_prepare_data.py<br/>
│   ├── svg_data_outputs/<br/>
│   │   ├── svg_tokenizer.json<br/>
│   │   ├── train_tokens.pt<br/>
│   │   ├── val_tokens.pt<br/>
│   │   ├── test_tokens.pt<br/>
│   │   └── metadata.json<br/>
│   └── svg_data_part4/   # same structure as svg_data_outputs<br/>
│<br/>
├── Colab Notebooks/<br/>
│   ├── part2.ipynb   # LR sweep, tiny models (SP)<br/>
│   ├── part3.ipynb   # scaling experiments, µP runs<br/>
│   └── part4.ipynb   # best model training, generation, evaluation<br/>

└── README.md
