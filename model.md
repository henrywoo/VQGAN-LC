# VQGAN Model

```python
🌳 VQModel<trainable_params:72421844,all_params:163938004,percentage:44.17636%>
├── Encoder(encoder)
│   ├── Conv2d(conv_in)|weight[128,3,3,3]|bias[128]
│   ├── ModuleList(down)
│   │   ├── 💠 Module(0-1)<🦜:738944x2>
│   │   │   ┣━━ ModuleList(block)
│   │   │   ┃   ┗━━ 💠 ResnetBlock(0-1)<🦜:295680x2>
│   │   │   ┃       ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:256x2>|weight[128]|bias[128]
│   │   │   ┃       ┗━━ 💠 Conv2d(conv1,conv2)<🦜:147584x2>|weight[128,128,3,3]|bias[128]
│   │   │   ┗━━ Downsample(downsample)
│   │   │       ┗━━ Conv2d(conv)|weight[128,128,3,3]🇸 -(2, 2)|bias[128]🇸 -(2, 2)
│   │   ├── Module(2)
│   │   │   ├── ModuleList(block)
│   │   │   │   ├── ResnetBlock(0)
│   │   │   │   │   ├── GroupNorm(norm1)|weight[128]|bias[128]
│   │   │   │   │   ├── Conv2d(conv1)|weight[256,128,3,3]|bias[256]
│   │   │   │   │   ├── GroupNorm(norm2)|weight[256]|bias[256]
│   │   │   │   │   ├── Conv2d(conv2)|weight[256,256,3,3]|bias[256]
│   │   │   │   │   └── Conv2d(nin_shortcut)|weight[256,128,1,1]|bias[256]
│   │   │   │   └── ResnetBlock(1)
│   │   │   │       ├── 💠 
│   │   │   │       │   GroupNorm(norm1,norm2)<🦜:512x2>|weight[256]|bias[256]
│   │   │   │       └── 💠 
│   │   │   │           Conv2d(conv1,conv2)<🦜:590080x2>|weight[256,256,3,3]|bia
│   │   │   │           s[256]
│   │   │   └── Downsample(downsample)
│   │   │       └── Conv2d(conv)|weight[256,256,3,3]🇸 -(2, 2)|bias[256]🇸 -(2, 2)
│   │   ├── Module(3)
│   │   │   ├── ModuleList(block)
│   │   │   │   └── 💠 ResnetBlock(0-1)<🦜:1181184x2>
│   │   │   │       ┣━━ 💠 
│   │   │   │       ┃   GroupNorm(norm1,norm2)<🦜:512x2>|weight[256]|bias[256]
│   │   │   │       ┗━━ 💠 
│   │   │   │           Conv2d(conv1,conv2)<🦜:590080x2>|weight[256,256,3,3]|bia
│   │   │   │           s[256]
│   │   │   └── Downsample(downsample)
│   │   │       └── Conv2d(conv)|weight[256,256,3,3]🇸 -(2, 2)|bias[256]🇸 -(2, 2)
│   │   └── Module(4)
│   │       ├── ModuleList(block)
│   │       │   ├── ResnetBlock(0)
│   │       │   │   ├── GroupNorm(norm1)|weight[256]|bias[256]
│   │       │   │   ├── Conv2d(conv1)|weight[512,256,3,3]|bias[512]
│   │       │   │   ├── GroupNorm(norm2)|weight[512]|bias[512]
│   │       │   │   ├── Conv2d(conv2)|weight[512,512,3,3]|bias[512]
│   │       │   │   └── Conv2d(nin_shortcut)|weight[512,256,1,1]|bias[512]
│   │       │   └── ResnetBlock(1)
│   │       │       ├── 💠 
│   │       │       │   GroupNorm(norm1,norm2)<🦜:1024x2>|weight[512]|bias[512]
│   │       │       └── 💠 
│   │       │           Conv2d(conv1,conv2)<🦜:2359808x2>|weight[512,512,3,3]|bi
│   │       │           as[512]
│   │       └── ModuleList(attn)
│   │           └── 💠 AttnBlock(0-1)<🦜:1051648x2>
│   │               ┣━━ GroupNorm(norm)|weight[512]|bias[512]
│   │               ┗━━ 💠 
│   │                   Conv2d(q,k,v,proj_out)<🦜:262656x4>|weight[512,512,1,1]|
│   │                   bias[512]
│   ├── Module(mid)
│   │   ├── 💠 ResnetBlock(block_1,block_2)<🦜:4721664x2>
│   │   │   ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:1024x2>|weight[512]|bias[512]
│   │   │   ┗━━ 💠 
│   │   │       Conv2d(conv1,conv2)<🦜:2359808x2>|weight[512,512,3,3]|bias[512]
│   │   └── AttnBlock(attn_1)
│   │       ├── GroupNorm(norm)|weight[512]|bias[512]
│   │       └── 💠 
│   │           Conv2d(q,k,v,proj_out)<🦜:262656x4>|weight[512,512,1,1]|bias[512
│   │           ]
│   ├── GroupNorm(norm_out)|weight[512]|bias[512]
│   └── Conv2d(conv_out)|weight[256,512,3,3]|bias[256]
├── Decoder(decoder)
│   ├── Conv2d(conv_in)|weight[512,256,3,3]|bias[512]
│   ├── Module(mid)
│   │   ├── 💠 ResnetBlock(block_1,block_2)<🦜:4721664x2>
│   │   │   ┣━━ 💠 GroupNorm(norm1,norm2)<🦜:1024x2>|weight[512]|bias[512]
│   │   │   ┗━━ 💠 
│   │   │       Conv2d(conv1,conv2)<🦜:2359808x2>|weight[512,512,3,3]|bias[512]
│   │   └── AttnBlock(attn_1)
│   │       ├── GroupNorm(norm)|weight[512]|bias[512]
│   │       └── 💠 
│   │           Conv2d(q,k,v,proj_out)<🦜:262656x4>|weight[512,512,1,1]|bias[512
│   │           ]
│   ├── ModuleList(up)
│   │   ├── Module(0)
│   │   │   └── ModuleList(block)
│   │   │       └── 💠 ResnetBlock(0-2)<🦜:295680x3>
│   │   │           ┣━━ 💠 
│   │   │           ┃   GroupNorm(norm1,norm2)<🦜:256x2>|weight[128]|bias[128]
│   │   │           ┗━━ 💠 
│   │   │               Conv2d(conv1,conv2)<🦜:147584x2>|weight[128,128,3,3]|bia
│   │   │               s[128]
│   │   ├── Module(1)
│   │   │   ├── ModuleList(block)
│   │   │   │   ├── ResnetBlock(0)
│   │   │   │   │   ├── GroupNorm(norm1)|weight[256]|bias[256]
│   │   │   │   │   ├── Conv2d(conv1)|weight[128,256,3,3]|bias[128]
│   │   │   │   │   ├── GroupNorm(norm2)|weight[128]|bias[128]
│   │   │   │   │   ├── Conv2d(conv2)|weight[128,128,3,3]|bias[128]
│   │   │   │   │   └── Conv2d(nin_shortcut)|weight[128,256,1,1]|bias[128]
│   │   │   │   └── 💠 ResnetBlock(1-2)<🦜:295680x2>
│   │   │   │       ┣━━ 💠 
│   │   │   │       ┃   GroupNorm(norm1,norm2)<🦜:256x2>|weight[128]|bias[128]
│   │   │   │       ┗━━ 💠 
│   │   │   │           Conv2d(conv1,conv2)<🦜:147584x2>|weight[128,128,3,3]|bia
│   │   │   │           s[128]
│   │   │   └── Upsample(upsample)
│   │   │       └── Conv2d(conv)|weight[128,128,3,3]|bias[128]
│   │   ├── Module(2)
│   │   │   ├── ModuleList(block)
│   │   │   │   └── 💠 ResnetBlock(0-2)<🦜:1181184x3>
│   │   │   │       ┣━━ 💠 
│   │   │   │       ┃   GroupNorm(norm1,norm2)<🦜:512x2>|weight[256]|bias[256]
│   │   │   │       ┗━━ 💠 
│   │   │   │           Conv2d(conv1,conv2)<🦜:590080x2>|weight[256,256,3,3]|bia
│   │   │   │           s[256]
│   │   │   └── Upsample(upsample)
│   │   │       └── Conv2d(conv)|weight[256,256,3,3]|bias[256]
│   │   ├── Module(3)
│   │   │   ├── ModuleList(block)
│   │   │   │   ├── ResnetBlock(0)
│   │   │   │   │   ├── GroupNorm(norm1)|weight[512]|bias[512]
│   │   │   │   │   ├── Conv2d(conv1)|weight[256,512,3,3]|bias[256]
│   │   │   │   │   ├── GroupNorm(norm2)|weight[256]|bias[256]
│   │   │   │   │   ├── Conv2d(conv2)|weight[256,256,3,3]|bias[256]
│   │   │   │   │   └── Conv2d(nin_shortcut)|weight[256,512,1,1]|bias[256]
│   │   │   │   └── 💠 ResnetBlock(1-2)<🦜:1181184x2>
│   │   │   │       ┣━━ 💠 
│   │   │   │       ┃   GroupNorm(norm1,norm2)<🦜:512x2>|weight[256]|bias[256]
│   │   │   │       ┗━━ 💠 
│   │   │   │           Conv2d(conv1,conv2)<🦜:590080x2>|weight[256,256,3,3]|bia
│   │   │   │           s[256]
│   │   │   └── Upsample(upsample)
│   │   │       └── Conv2d(conv)|weight[256,256,3,3]|bias[256]
│   │   └── Module(4)
│   │       ├── ModuleList(block)
│   │       │   └── 💠 ResnetBlock(0-2)<🦜:4721664x3>
│   │       │       ┣━━ 💠 
│   │       │       ┃   GroupNorm(norm1,norm2)<🦜:1024x2>|weight[512]|bias[512]
│   │       │       ┗━━ 💠 
│   │       │           Conv2d(conv1,conv2)<🦜:2359808x2>|weight[512,512,3,3]|bi
│   │       │           as[512]
│   │       ├── ModuleList(attn)
│   │       │   └── 💠 AttnBlock(0-2)<🦜:1051648x3>
│   │       │       ┣━━ GroupNorm(norm)|weight[512]|bias[512]
│   │       │       ┗━━ 💠 
│   │       │           Conv2d(q,k,v,proj_out)<🦜:262656x4>|weight[512,512,1,1]|
│   │       │           bias[512]
│   │       └── Upsample(upsample)
│   │           └── Conv2d(conv)|weight[512,512,3,3]|bias[512]
│   ├── GroupNorm(norm_out)|weight[128]|bias[128]
│   └── Conv2d(conv_out)|weight[3,128,3,3]|bias[3]
├── NLayerDiscriminator(discriminator)
│   └── Sequential(main)
│       ├── Conv2d(0)|weight[64,3,4,4]🇸 -(2, 2)|bias[64]🇸 -(2, 2)
│       ├── Conv2d(2)|weight[128,64,4,4]🇸 -(2, 2)
│       ├── BatchNorm2d(3)|weight[128]|bias[128]
│       ├── Conv2d(5)|weight[256,128,4,4]
│       ├── BatchNorm2d(6)|weight[256]|bias[256]
│       └── Conv2d(8)|weight[1,256,4,4]|bias[1]
├── LPIPS(perceptual_loss)
│   ├── vgg16(net)
│   │   ├── Sequential(slice1)
│   │   │   ├── Conv2d(0)|weight[64,3,3,3]❄️|bias[64]❄️
│   │   │   └── Conv2d(2)|weight[64,64,3,3]❄️|bias[64]❄️
│   │   ├── Sequential(slice2)
│   │   │   ├── Conv2d(5)|weight[128,64,3,3]❄️|bias[128]❄️
│   │   │   └── Conv2d(7)|weight[128,128,3,3]❄️|bias[128]❄️
│   │   ├── Sequential(slice3)
│   │   │   ├── Conv2d(10)|weight[256,128,3,3]❄️|bias[256]❄️
│   │   │   └── 💠 
│   │   │       Conv2d(12-12,14-14)<🦜:0,590080x2>|weight[256,256,3,3]❄️|bias[256
│   │   │       ]❄️
│   │   ├── Sequential(slice4)
│   │   │   ├── Conv2d(17)|weight[512,256,3,3]❄️|bias[512]❄️
│   │   │   └── 💠 
│   │   │       Conv2d(19-19,21-21)<🦜:0,2359808x2>|weight[512,512,3,3]❄️|bias[51
│   │   │       2]❄️
│   │   └── Sequential(slice5)
│   │       └── 💠 
│   │           Conv2d(24-24,26-26,28-28)<🦜:0,2359808x3>|weight[512,512,3,3]❄️|b
│   │           ias[512]❄️
│   ├── NetLinLayer(lin0)
│   │   └── Sequential(model)
│   │       └── Conv2d(1)|weight[1,64,1,1]❄️
│   ├── NetLinLayer(lin1)
│   │   └── Sequential(model)
│   │       └── Conv2d(1)|weight[1,128,1,1]❄️
│   ├── NetLinLayer(lin2)
│   │   └── Sequential(model)
│   │       └── Conv2d(1)|weight[1,256,1,1]❄️
│   └── 💠 NetLinLayer(lin3,lin4)<🦜:0,512x2>
│       ┗━━ Sequential(model)
│           ┗━━ Conv2d(1)|weight[1,512,1,1]❄️
├── Embedding(tok_embeddings)|weight[100000,768]❄️
├── Conv2d(quant_conv)|weight[8,256,1,1]|bias[8]
├── Conv2d(post_quant_conv)|weight[256,8,1,1]|bias[256]
└── Linear(codebook_projection)|weight[8,768]|bias[8]
```