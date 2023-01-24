import torch
import setuptools
import transformers as hf
import limber_gptj
from torch.optim import AdamW
from collections import defaultdict
import sys
from dataset import EncCptDataset,ImgCptDataset
from torch.utils.data import DataLoader
from config_params import get_params_for_weight_decay_optimization, configure_param_groups

from torch.optim.lr_scheduler import LambdaLR
import math

from dataclasses import dataclass
from transformers import GPT2TokenizerFast
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy

from deepspeed.runtime.lr_schedules import WarmupLR, WarmupDecayLR
from transformers.modeling_utils import no_init_weights

import os

os.environ["WANDB_WATCH"] = 'false'


def conceptual_captions(data_dir,tokenizer,transforms, batch_size=256, num_workers=1):
    train_dataset = ImgCptDataset(data_dir,  tokenizer, transforms)
    return train_dataset


@dataclass
class ImageCaptionDataCollator:
    tokenizer: GPT2TokenizerFast
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        #print(features, 'features')
        batch = {}
        batch['input_ids']=[f["captions"] for f in features]
        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch['captions']=batch['input_ids']
        batch['images'] = torch.stack([i['images'] for i in features]).float()
        batch['attention_mask'] = torch.stack([torch.cat( (torch.ones(config.image_seq_len), a) ) for a in batch['attention_mask']]).float() #prepend 144 ones at beginning
        del batch['input_ids']
        return batch

seed = 0
print("DEVICE COUNT", torch.cuda.device_count())
hf.set_seed(seed)
print("WORLd SIZE", os.environ["WORLD_SIZE"])
print("Creating model")
with no_init_weights():
    lm = limber_gptj.LimberGPTJ.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
print("Loaded model")



config_file = "configs/beit_random_linear.yml" 



exp_name = config_file.split("/")[1].split('.')[0]
lm.setup_multimodal(config_file)

config = lm.multimodal_config
print("THEDEVICE", lm.device)
tokenizer = lm.tokenizer
transforms = lm.transforms
print("Setup multimodal")
print("Zero stage", config.zero_stage)

config.gradient_accumulation_steps=config.gradient_accumulation_steps
config.deepspeed_config_params["gradient_accumulation_steps"]=config.deepspeed_config_params["gradient_accumulation_steps"]
print("GRADIENT_ACCUMULATION", config.gradient_accumulation_steps, config.deepspeed_config_params["gradient_accumulation_steps"])



adjusted_train_batch_size = config.batch_size//int(os.environ["WORLD_SIZE"])

print("BATCH SIZE", adjusted_train_batch_size)
train_data = conceptual_captions(config.train_dataset_dir, tokenizer, transforms, batch_size=config.batch_size)
trainable_parameters = configure_param_groups(lm, lm.multimodal_config)

opt = AdamW(
    trainable_parameters,#
    lm.multimodal_config.lr,
    betas=(0.9, 0.95),
    weight_decay=lm.multimodal_config.weight_decay,
)

scheduler = WarmupDecayLR(opt, config.lr_decay_iters)
print("THIS MANY TRAINING STEPS:", config.train_steps)
train_args = hf.TrainingArguments(
        output_dir=exp_name,
        overwrite_output_dir=True,
        evaluation_strategy='no',
        eval_steps=50,
        per_device_train_batch_size=adjusted_train_batch_size,
        per_device_eval_batch_size=adjusted_train_batch_size,
        dataloader_drop_last=True, #Drop last batch if not divisible by batch size
        bf16=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_num_steps,
        max_steps=config.train_steps,
        adam_beta1=.9,
        adam_beta2=.95,
        weight_decay=config.weight_decay,
        logging_steps= 1,
        learning_rate=[0.0008],
        remove_unused_columns=True,
        dataloader_num_workers=1, #TODO set as argument
        disable_tqdm=True,
        log_level='error',
        save_strategy='steps',
        save_steps=500,
        save_total_limit=1,
        seed=42,
        deepspeed= config.deepspeed_config_params if torch.cuda.device_count()>1 else None #config.deepspeed_config_params#'ds_zero2.json'
)

[print('Allocated before train:', round(torch.cuda.memory_allocated(i)/1024**3,1), 'GB') for i in range(torch.cuda.device_count())]

lm = lm.train()
lm.gradient_checkpointing_enable()

trainer = hf.Trainer(
    lm,
    train_args,
    train_dataset=train_data,
    tokenizer = tokenizer,
    optimizers = (opt, scheduler),
    data_collator=ImageCaptionDataCollator(tokenizer),
)
trainer.train()
print("DONE TRAINING::")


#Run with: deepspeed hf_main.py