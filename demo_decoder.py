import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.unets import UNet2DConditionModel
from hiq import print_model

model_id = "OFA-Sys/small-stable-diffusion-v0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Get the UNet model
unet_model = pipe.unet

# Print model details
print_model(unet_model)

prompt = "an apple, 4k"
image = pipe(prompt).images[0]

image.save("apple.png")
