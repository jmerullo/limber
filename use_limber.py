from image_input import ImageInput
import sys
import os
from limber_gptj import LimberGPTJ
import torch
import numpy as np
from PIL import Image

def simple_load_model(config_path, limber_proj_path='auto', device='cuda:0'):
    lm =LimberGPTJ.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.bfloat16)
    lm.setup_multimodal(config_path, device=device)
    if limber_proj_path == 'auto':
        if config_path.endswith("beit_ft_linear.yml"):
            limber_proj_path = 'limber_weights/beit_ft_linear/proj.ckpt'
        elif config_path.endswith("beit_linear.yml"):
            limber_proj_path = 'limber_weights/beit_linear/proj.ckpt'
        elif config_path.endswith("nfrn50_4096_linear.yml"):
            limber_proj_path = 'limber_weights/nfrn50_4096_linear/proj.ckpt'
        elif config_path.endswith('nfrn50_4096_random_linear.yml'):
            limber_proj_path = 'limber_weights/nfrn50_4096_linear/proj.ckpt'
        elif config_path.endswith('clip_linear.yml'):
            limber_proj_path = 'limber_weights/clip_linear/proj.ckpt'
    proj_ckpt = torch.load(limber_proj_path)
    lm.image_prefix.proj.load_state_dict(proj_ckpt) #Load in the weights for the linear projection
    return lm


if __name__ == "__main__":
    config_path = 'configs/clip_linear.yml'
    model = simple_load_model(config_path)
    print("Loaded model")
    model = model.cuda().half()
    #Example image from MAGMA repo:
    imginp = ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg')
    inputs = model.preprocess_inputs([imginp, 'A picture of'])
    output = model.generate(embeddings=inputs)
    print(output)
    #BEIT linear: [' a traditional house in the village']
    #NFRN50 linear: [' a mountain village in the mountains.']
    #CLIP linear: [' a house in the woods']
