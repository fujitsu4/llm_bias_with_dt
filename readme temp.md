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

## License
This project is licensed under the MIT License – see the LICENSE file for details.