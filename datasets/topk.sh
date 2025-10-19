DATA=/home/lym/VLM-MSA/datasets/$DATASET_NAME
TAG=sbert-roberta-large

python precompute_topk.py \
  --train_emb $DATA/train_${TAG}.npy \
  --query_emb $DATA/${MODE}_${TAG}.npy \
  --store_k 10 \
  --out_prefix $DATA/${MODE}2train_${TAG} \
  --train_file $DATA/train_few1.json \
  --label_field  label \
  --dataset_name $DATASET_NAME \
  --per_class_k  5 \
  --balance_strategy roundrobin
