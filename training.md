# My test



- DS: DS_PATH_OXFLOWER_7K

It seems rFID is smaller when dataset is bigger.

```python
[22:47:51.677735] Averaged stats: lpips: 0.1048 (0.1085)  psnr: 25.6718 (25.1300)  ssim: 0.6595 (0.6382)
[22:47:51.677992] FID: 8.90229799396701
[22:47:51.678037] LPIPS: 0.1047778156848279
[22:47:51.678065] PSNR: 25.682879411354442
[22:47:51.678088] SSIM: 0.6596411333737272
[22:47:51.679449] Effective Tokens: 98857
```

- GPT Generation

```python
python vqgan-gpt-lc/eval_generation.py \
--batch_size 8 --image_size 256 --epochs 100 --n_class 1000 --num_workers 8 \
--vq_config_path vqgan-gpt-lc/vqgan_configs/vq-f16.yaml \
--output_dir "log_eval_gpt/gpt_lc_100K_f16" --local_embedding_path mbin/codebook-100K.pth \
--stage_1_ckpt mbin/vqgan-lc-100K-f16-dim8.pth --stage_2_ckpt mbin/gpt-lc-100K-f16.pth \
--n_vision_words 100000 --tuning_codebook 0 --use_cblinear 1 --embed_dim 8 \
--top_k 100000 --dataset "imagenet" --gpt_type "small"
```

