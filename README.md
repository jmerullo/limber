# LIMBER: Linearly Mapping Between Representation spaces

**Paper:** https://arxiv.org/abs/2209.15162

**Authors:** Jack Merullo, Louis Castricato, Carsten Eickhoff, Ellie Pavlick

Limber models are trained by stitching an image encoder (CLIP, NF-Resnet50, BEIT) to a text decoder (GPT-J) with a linear layer. This repo provides code for training and using limber models as they are presented in the paper.

![Model](examples/model_arch.png?raw=true "Limber model")


## Requirements
Python >=3.7.4 
First, run `pip install numpy torch==1.13.0 torchvision https://github.com/openai/CLIP/archive/master.zip` before install the other dependencies with `pip install -r requirements.txt`

## Inference
Limber models are useful tools for interpreting multimodal and text-only models, or as a baseline for image captioning systems. We provide code and the linear projection weights needed for using these models. The weights for the linear projections are provided as well in the `limber_weights` directory. `use_limber.py` provides sample code for generating a caption for a sample image:
```
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
```

The LimberGPTJ class subclasses the "EleutherAI/gpt-j-6B" huggingface model. To setup the model to accept images and text as input, pass the path of a config file to the `setup_multimodal` method. This method will instantiate the image prefix used to encode images. Once embeddings are created, they can be passed into the model's generate method to create a caption. The underlying GPT-J model are completely unchanged from their default state when used this way. Note that some layers in image encoders are changed for convenience but are otherwise frozen, e.g., the final CLIP attention pooling layer is turned into the identity matrix to simplify extracting the required hidden representations (this mimics the way [MAGMA](https://github.com/Aleph-Alpha/magma) is implemented).

## Training
The code used to train the linear projections stitching the image and language models together can be found in `hf_main.py`. Training can be replicated with the provided `configs` and running `deepspeed hf_main.py`. The models used in the paper were trained on 16 40GB A100 gpus for 15,000 training steps on the CC3M image captions dataset. 
