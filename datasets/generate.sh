
# pip install -U sentence-transformers
python generate_emb.py \
  --jsonl /home/lym/VLM-MSA/datasets/$DATASET_NAME/train_few1.json \
  --data_dir /home/lym/VLM-MSA/datasets/$DATASET_NAME \
  --split train \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large

python generate_emb.py \
  --jsonl /home/lym/VLM-MSA/datasets/$DATASET_NAME/dev_few1.json \
  --data_dir /home/lym/VLM-MSA/datasets/$DATASET_NAME \
  --split val \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large

python generate_emb.py \
  --jsonl /home/lym/VLM-MSA/datasets/$DATASET_NAME/test.json \
  --data_dir /home/lym/VLM-MSA/datasets/$DATASET_NAME \
  --split test \
  --hf_model /home/lym/MultiPoint/models/sbert-roberta-large
