DATA=/home/lym/VLM-MSA/datasets/$DATASET_NAME
TAG=sbert-roberta-large

python precompute_topk.py \
  --train_emb $DATA/train_${TAG}.npy \
  --query_emb $DATA/${MODE}_${TAG}.npy \
  --store_k 10 \
  --out_prefix $DATA/${MODE}2train_${TAG} \
  --save_sims
