# llm_bias_with_dt

This project contains scripts used in the article **"Exploring biases in BERT through attention scores: a Decision Tree"***.

## A.Project structure
llm_bias_with_dt/
│
├── src/ # All Python scripts (with submodules)
│ ├── prepare/
│ │ ├── prepare_snli.py
│ │ ├── prepare_mnli.py
│ │ └── merge_datasets.py
│ │
│ ├── spacy/
│ │ ├── compute_spacy_features.py
│ │ └── verify_spacy_features.py
│ │
│ └── utils/ # Small functions Auxiliaries if needed
│
├── data/
│ ├── raw/ # possibly to store the original datasets
│ ├── cleaned/ # snli_filtered_sentences.csv, mnli_filtered_sentences.csv, merged_sentences.csv
│ └── features/ # merged_spacy_features.csv, spacy_label_maps.json
│
├── logs/
│ ├── snli_rejected.txt
│ ├── mnli_rejected.txt
│ └── ... (optional)
│
├── README.md # Project documentation
├── requirements.txt # Dependencies
├── LICENSE # MIT if you want
└── main.py # OPTIONAL: Orchestrator pipeline (optional)
├── pyproject.toml #(OPTIONAL, to declare a real Python package)

## B.Installation

```bash
git clone https://github.com/USER/llm_bias_with_dt
cd llm_bias_with_dt

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## C.How to Run
1. Prepare datasets
python src/prepare/prepare_dataset_snli.py
python src/prepare/prepare_dataset_mnli.py

2. Merge
python src/merge/merge_datasets.py

3. Compute linguistic features
python src/features/compute_spacy_features.py

4. Verify integrity
python src/features/verify_spacy_features.py

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