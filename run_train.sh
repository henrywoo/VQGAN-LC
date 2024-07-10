python vqgan-gpt-lc/training_vqgan.py \
--batch_size 4 --image_size 256 --epochs 100 --warmup_epochs 5 --lr 5e-4 --n_class 1000 \
--num_workers 16 --vq_config_path vqgan-gpt-lc/vqgan_configs/vq-f16.yaml \
--output_dir "train_logs_vq/vqgan_lc_100K" --log_dir "train_logs_vq/vqgan_lc_100K" \
--disc_start 50000 --n_vision_words 100000 --local_embedding_path mbin/codebook-100K.pth \
--tuning_codebook 0 --use_cblinear 1 --embed_dim 8