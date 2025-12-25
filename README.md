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

## C.Reproducibility Pipeline

This section describes the complete experimental pipeline used to reproduce the results reported in the paper. All commands are executed from the project root directory.

The pipeline is divided into **four main stages**, followed by **two execution branches** corresponding to **pretrained and untrained BERT** models. Unless stated otherwise, each processing step is executed once and shared by both branches.

**1. Dataset Preparation**

Four datasets (SNLI, MNLI, ArXiv, AGNews) are filtered and subsampled to ensure balanced and controlled inputs. Each dataset is processed independently, then merged into a single dataset used throughout the pipeline.
```
python -m src.prepare.prepare_dataset_snli --output data/cleaned/snli_filtered.csv --target 2500
python -m src.prepare.prepare_dataset_mnli --output data/cleaned/mnli_filtered.csv --target 2500
python -m src.prepare.prepare_dataset_arxiv --output data/cleaned/arxiv_filtered.csv --target 2500
python -m src.prepare.prepare_dataset_agnews --output data/cleaned/agnews_filtered.csv --target 2500

python -m src.prepare.merge_datasets \
    --inputs data/cleaned/snli_filtered.csv \
             data/cleaned/mnli_filtered.csv \
             data/cleaned/arxiv_filtered.csv \
             data/cleaned/agnews_filtered.csv \
    --output data/cleaned/merged_datasets.csv
```

**2. Linguistic Feature Extraction (SpaCy)**

Linguistic features are extracted using SpaCy and verified for consistency against the original sentences.
```
python -m src.spacy.compute_spacy_features \
    --input data/cleaned/merged_datasets.csv \
    --output outputs/spacy/spacy_features.csv

python -m src.spacy.verify_spacy_features \
    --sentences_csv data/cleaned/merged_datasets.csv \
    --features_csv outputs/spacy/spacy_features.csv \
    --log_file logs/spacy_logs.txt
```
**3. BERT Tokenization and Feature Construction**

Tokens are aligned between SpaCy and BERT, followed by the extraction of basic, statistical, and neighborhood-based features. Each step includes an explicit verification script.
```
python -m src.bert.tokenize_with_bert \
    --sentences_csv data/cleaned/merged_datasets.csv \
    --spacy_csv outputs/spacy/spacy_features.csv \
    --output outputs/bert/bert_tokens.csv

python -m src.bert.verify_bert_tokenization \
    --bert_tokens_csv outputs/bert/bert_tokens.csv \
    --spacy_csv outputs/spacy/spacy_features.csv \
    --log_file logs/bert_tokenization_logs.txt

python -m src.bert.compute_bert_basic_features \
    --bert_csv outputs/bert/bert_tokens.csv \
    --output outputs/bert/bert_basic_features.csv

python -m src.bert.verify_bert_basic_features \
    --features_csv outputs/bert/bert_basic_features.csv \
    --log_file logs/bert_basic_features_logs.txt

python -m src.bert.compute_bert_statistical_features \
    --bert_features outputs/bert/bert_basic_features.csv \
    --output outputs/bert/bert_statistical_features.csv

python -m src.bert.verify_bert_statistical_features \
    --features_csv outputs/bert/bert_statistical_features.csv \
    --log_file logs/bert_statistical_features_logs.txt

python -m src.bert.merge_spacy_and_bert \
    --spacy_csv outputs/spacy/spacy_features.csv \
    --bert_csv outputs/bert/bert_statistical_features.csv \
    --output outputs/bert/spacy_bert_merged.csv

python -m src.bert.compute_neighbor_pos_features \
    --input outputs/bert/spacy_bert_merged.csv \
    --output outputs/bert/bert_final_features.csv

python -m src.bert.verify_final_features \
    --input outputs/bert/bert_final_features.csv \
    --log logs/bert_final_logs.txt
```
**4. Attention Extraction (Pretrained vs Untrained)**

*4.1 Pretrained BERT*

Attention scores are computed once using the pretrained BERT model.
```
python -m src.attention.compute_attention_top5 \
    --model pretrained \
    --input_csv outputs/bert/bert_final_features.csv
```
*4.2 Untrained BERT (Multiple Seeds)*

A fixed list of random seeds is generated once, then used to compute attention scores independently for each untrained model initialization.
```
python -m src.seeds.generate_seeds \
    --count 30 --low 1 --high 10000 \
    --output outputs/attention/seeds_list.txt
```

Attention extraction is then executed per seed (typically via a shell script for automation):

```
bash run_attention_seeds.sh outputs/attention/seeds_list.txt
```

**5. Decision Tree Training and Verification**

*5.1 Pretrained BERT*

Decision trees are trained independently for each layer (12 layers) using the extracted attention scores. This step is performed separately for pretrained and untrained models.
```
python -m src.decision_tree.compute_decision_trees \
    --model pretrained \
    --input_csv outputs/attention/attention_top5_pretrained.csv
```

*5.2 Untrained BERT (Multiple Seeds)*

Decision tree are computed per seed (typically via a shell script for automation):

```
bash run_decision_tree_seeds.sh /content/llm_bias_with_dt/outputs/attention/seeds_list.txt --max-runs 3
```

Verification scripts ensure tree reproducibility and consistency across seeds and layers.

**6. Decision Tree Analysis**

Extracted decision trees are transformed into interpretable statistics and patterns.

*6.1 Feature Usage Analysis*

```
python -m src.dt_analysis.extract_dt_statistics \
    --input_dir outputs/decision_tree/pretrained

python -m src.dt_analysis.aggregate_features_by_depth \
    --mode pretrained \
    --pretrained_csv outputs/dt_analysis/dt_features_depth_pretrained.csv \
    --output_csv outputs/dt_analysis/features_by_depth_pretrained.csv
```

Equivalent commands are applied to the untrained directory.

*6.2 Missing Feature Analysis*

```
python -m src.dt_analysis.compute_missing_features \
    --attention_csv outputs/attention/attention_top5_pretrained_sample.csv \
    --pretrained_csv outputs/dt_analysis/features_by_depth_pretrained.csv \
    --untrained_csv outputs/dt_analysis/features_by_depth_untrained.csv \
    --log_file logs/missing_features_comparison.txt
```

*6.3 Pattern Aggregation*

```
python -m src.dt_analysis.aggregate_features_patterns \
    --input_dir outputs/decision_tree/pretrained \
    --output_csv outputs/dt_analysis/features_patterns_pretrained.csv
```

And equivalently for untrained models.

**Optional: Debug and Visualization**

Several debug and visualization scripts are provided to inspect attention maps and individual decision trees. These scripts are not required for reproduction, but are included for transparency and qualitative inspection.

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