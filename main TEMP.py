python -m src.prepare.prepare_dataset_snli --output data/cleaned/snli_filtered.csv --target 2500
python -m src.prepare.prepare_dataset_mnli --output data/cleaned/mnli_filtered.csv --target 2500

python src/merge/merge_datasets.py
python src/features/compute_spacy_features.py
python src/features/verify_spacy_features.py
