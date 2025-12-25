# llm_bias_with_dt

## A.Project overview
This project contains scripts used in the article **"Exploring Biases in BERT through Attention Scores: A Decision Tree Approach"**.
This is a research-oriented framework designed to analyze and characterize bias-related decision patterns in BERT models using interpretable decision trees.

The project follows a structured pipeline that:

1. Clean and prepare datasets,
2. Extracts attention scores from pretrained and untrained BERT models,
3. Derives linguistic and statistical token-level features,
4. Trains decision trees at each transformer layer,
5. Analyzes stable feature usage and compositional decision patterns across layers and random seeds.

Particular emphasis is placed on **interpretability**, **reproducibility**, and **robustness across random initializations**.

All major processing steps are accompanied by dedicated verification scripts to ensure the correctness of intermediate and final results.

The framework supports both **pretrained** BERT and **untrained** BERT models, enabling a systematic comparison between emergent structures induced by training and architectural biases.

## B.Project structure
```
LLM_BIAS_WITH_DT/
├── data/
│   └── cleaned/                 # Filtered and merged datasets (AGNews, ArXiv, MNLI, SNLI)
│
├── src/
│   ├── attention/               # Attention extraction and verification
│   ├── bert/                    # BERT-based linguistic and statistical features
│   ├── spacy/                   # SpaCy feature extraction
│   ├── decision_tree/           # Decision tree training and validation
│   ├── dt_analysis/             # Decision tree analysis and aggregation
│   ├── prepare/                 # Dataset preparation scripts
│   ├── seeds/                   # Seed generation utilities
│   ├── utils/                   # Shared utilities
│
├── outputs/
│   ├── attention/               # Attention scores (samples + seeds list)
│   ├── bert/                    # Extracted BERT features (samples)
│   ├── decision_tree/           # Decision tree rules (pretrained / untrained)
│   ├── dt_analysis/             # Final aggregated results (CSV + ZIP)
│   └── spacy/                   # SpaCy feature outputs
│
├── logs/                        # Execution and verification logs
│
├── run_attention_seeds.sh       # Batch execution for attention extraction
├── run_decision_tree_seeds.sh   # Batch execution for decision tree training
│
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
```

## B.Installation

This project was developed and tested with Python ≥ 3.10.

**1. Clone the repository**
```
git clone https://github.com/fujitsu4/llm_bias_with_dt.git
cd llm_bias_with_dt
```

**2. Create a virtual environment (recommended)**
```
python -m venv venv
source venv/bin/activate  # Linux / macOS
# venv\Scripts\activate   # Windows
```

**3. Install Python dependencies**
```
pip install -r requirements.txt
```

**4. Download required SpaCy language model**
```
python -m spacy download en_core_web_sm
```

**GPU support**

To use GPU, ensure that the installed PyTorch version is compatible with your CUDA version.

**External Resources and Caching**

Some resources are downloaded automatically at runtime:
* HuggingFace models and tokenizers (e.g., BERT)
* HuggingFace datasets

These resources are cached locally (typically under ~/.cache/) and are downloaded only once.
No manual download is required for these components.

## 📁 Data Availability and Repository Structure

To keep this repository lightweight, readable, and compliant with GitHub’s recommended size constraints, **only lightweight sample files of the intermediate outputs are included here**.  
These sample files (5 lines each) are provided exclusively for **illustration, documentation, and structural transparency**.

---

### 🔹 Why only sample files?

Some intermediate CSV files generated during the experiments (e.g., BERT token features, attention matrices, statistical descriptors) originally exceed **30–100 MB per file**, and the complete pipeline produces **hundreds of such files**.  
Including them directly in the repository would:

- inflate the repository size unnecessarily,  
- violate large file best practices,  
- slow down cloning and CI workflows,  
- reduce overall readability for users and reviewers.

Therefore, this repository includes files such as:

bert_final_features_SAMPLE.csv
bert_basic_features_SAMPLE.csv
spacy_features_SAMPLE.csv
attention_top5_pretrained_SAMPLE.csv


Each sample keeps **only the first 5 rows**, allowing readers to understand:

- the file schema,  
- the column definitions,  
- the preprocessing workflow,  
- and how each component interacts in the pipeline,  

without including full, heavy intermediate outputs.

---

## 📦 Full Reproducibility

The **complete intermediate results**, including all intermediate and large output files, are archived externally (Google Drive) to ensure:

- long-term preservation,  
- stable access,  
- citable references,  
- compliance with FAIR principles.  

To reproduce the full pipeline, simply follow all the commands listed at the beginning of this readme. The code will place them automatically into the correct directories (without any truncation of the results).

---

## 📁 Debug Samples

The folder `outputs/debug_samples/` contains a minimal and lightweight subset of
debugging outputs used to illustrate the internal structure of the pipeline.
Only one complete example is kept for each setting (pretrained / untrained), for
documentation purposes. The full debug output originally generated during the
runs has been intentionally omitted to keep the repository clean and compact,
as it is not required for reproducibility or analysis.

---

## 📝 Notes

- All sample files follow the naming pattern `*_sample.csv` to clearly distinguish them from full data files.  
- The repository is intentionally structured to remain **fast to clone**, **easy to inspect**, and **fully reproducible** once external datasets are provided.
-The `data/cleaned` and `logs/` folders are kept in full and were not cleaned or reduced.
They contain the processed datasets and the complete processing logs required to ensure full reproducibility and transparency of the experiments.

---

## License
This project is licensed under the MIT License – see the LICENSE file for details.