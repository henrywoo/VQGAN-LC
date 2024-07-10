python vqgan-gpt-lc/eval_reconstruction.py \
--batch_size 24 --image_size 256 --lr 9e-3 --n_class 1000 \
--vq_config_path vqgan-gpt-lc/vqgan_configs/vq-f16.yaml --output_dir "log_eval_recons/vqgan_lc_100K_f16" \
--log_dir "log_eval_recons/vqgan_lc_100K_f16" --quantizer_type "org" \
--local_embedding_path mbin/codebook-100K.pth --stage_1_ckpt mbin/vqgan-lc-100K-f16-dim8.pth \
--tuning_codebook 0 --embed_dim 8 --n_vision_words 100000 --use_cblinear 1 --dataset "imagenet"