```python
🌳 SiT<trainable_params:674890048,all_params:674963776,percentage:99.98908%>
├── PatchEmbed(x_embedder)
│   └── Conv2d(proj)|weight[1152,8,2,2]🇸 -(2, 2)|bias[1152]🇸 -(2, 2)
├── TimestepEmbedder(t_embedder)
│   └── Sequential(mlp)
│       ├── Linear(0)|weight[1152,256]|bias[1152]
│       └── Linear(2)|weight[1152,1152]|bias[1152]
├── LabelEmbedder(y_embedder)
│   └── Embedding(embedding_table)|weight[1001,1152]
├── ModuleList(blocks)
│   └── 💠 SiTBlock(0-27)<🦜:23905152x28>
│       ┣━━ Attention(attn)
│       ┃   ┣━━ Linear(qkv)|weight[3456,1152]|bias[3456]
│       ┃   ┗━━ Linear(proj)|weight[1152,1152]|bias[1152]
│       ┣━━ Mlp(mlp)
│       ┃   ┣━━ Linear(fc1)|weight[4608,1152]|bias[4608]
│       ┃   ┗━━ Linear(fc2)|weight[1152,4608]|bias[1152]
│       ┗━━ Sequential(adaLN_modulation)
│           ┗━━ Linear(1)|weight[6912,1152]|bias[6912]
└── FinalLayer(final_layer)
    ├── Linear(linear)|weight[64,1152]|bias[64]
    └── Sequential(adaLN_modulation)
        └── Linear(1)|weight[2304,1152]|bias[2304]
```