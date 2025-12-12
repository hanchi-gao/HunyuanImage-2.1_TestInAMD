import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
# Supported model_name: hunyuanimage-v2.1,
# hunyuanimage-v2.1-distilled
model_name = "hunyuanimage-v2.1"
pipe = HunyuanImagePipeline.from_pretrained(
 model_name=model_name,
 use_fp8=True
)
# 手动分配模型的不同部分到不同GPU
if hasattr(pipe, 'text_encoder'):
    pipe.text_encoder = pipe.text_encoder.to("cuda:0")
if hasattr(pipe, 'tokenizer_2'):
    pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda:0")
if hasattr(pipe, 'unet'):
    pipe.unet = pipe.unet.to("cuda:1")
if hasattr(pipe, 'vae'):
    pipe.vae = pipe.vae.to("cuda:0")
#pipe = pipe.to("cuda")
#The input prompt
prompt = (
 "A cute, cartoon-style anthropomorphic penguin plush toy "
 "with fluffy fur, standing in a painting studio, wearing "
 "a red knitted scarf and a red beret with the word "
 "\"Asrock\" on it, holding a paintbrush with a focused "
 "expression as it paints an oil painting of the Mona Lisa, "
 "rendered in a photorealistic photographic style."
)
# Generate with different aspect ratios
aspect_ratios = {
 "16:9": (2560, 1536),
 "4:3": (2304, 1792),
 "1:1": (2048, 2048),
 "3:4": (1792, 2304),
 "9:16": (1536, 2560),
}
width, height = aspect_ratios["1:1"]
image = pipe(
 prompt=prompt,
 width=width,
 height=height,
 use_reprompt=False, # Disable reprompt (save memory)
 use_refiner=True, # Disable refiner (avoid VRAM shortage)
 num_inference_steps=8 if "distilled" in model_name else 50,
 guidance_scale=3.25 if "distilled" in model_name else 3.5,
 shift=4 if "distilled" in model_name else 5,
 seed=649151,
)
image.save("generated_image.png")
