# This file was adapted from the MAGMA repo:
# https://github.com/Aleph-Alpha/magma/blob/master/magma/datasets/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL.Image import Image as img
from PIL.Image import DecompressionBombError
from PIL import UnidentifiedImageError
import json
from pathlib import Path

from tqdm import tqdm
from typing import List, Tuple, Generator
import random
from multiprocessing import Pool, cpu_count

from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple
from torchtyping import TensorType
import traceback

import numpy as np

def read_jsonl(filename: str) -> Generator[List, None, None]:
    """
    Iterator over data from a jsonl file
    """
    with open(filename) as file:
        for line in file:
            yield json.loads(line.rstrip("\n|\r"))


def read_img_captions(filename: str) -> List[Tuple[str, str]]:
    """
    Yields image_path, image_caption from cc jsonl files
    """
    img_captions = []
    for item in read_jsonl(filename):
        if not "N/A" in item[-2:]:
            img_captions.append((item[-1], item[-2]))
    return img_captions


def load_json(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except Exception:
        print(f"ERROR: Error loading json file {filename}")
        traceback.print_exc()


def _bak_read_image_data(data_dir,subdir='*'):
    image_data = []
    img_data_dir = data_dir / "image_data"
    paths = _load_paths(data_dir, subdir=subdir)
    #print("PATHS[0]", paths[0])
    pbar = tqdm(
        paths,
        desc=f"loading dataset from {str(data_dir)}",
    )
    # read data with multiprocessing
    with Pool(cpu_count()) as pool:
        for img_data in pool.imap(load_json, pbar):
            if img_data is not None:
                image_data.append(img_data)
    return image_data

def _read_image_data(data_dir, subdir='*'):
    image_data = []
    img_data_dir = data_dir / "image_data"
    paths = _load_paths(data_dir, subdir=subdir)
    print('p0', paths[0])
    pbar = tqdm(
        paths,
        desc=f"loading dataset from {str(data_dir)}",
    )
    # read data with multiprocessing
    with Pool(cpu_count()) as pool:
        for img_data in pool.imap(load_json, pbar):
            if img_data is not None:
                image_data.append(img_data)
    return image_data

def _read_image_enc_data(data_dir, encs_path):
    encs_dir = os.path.join(data_dir, encs_path)
    img_dir = data_dir / 'images'
    enc_data = []
    #print('encd_dir', encs_dir)
    paths = Path(encs_dir).glob("*/*.npy") # Each one like: .../magma_viz_encodings/clip_resnet_large/1/12360_2803494085.npy
    for p in paths:
        pname = str(p)
        img_file = '/'.join(pname[:-4].split('/')[-2:]) #.../1/12360_2803494085.npy becomes 1/12360_2803494085
        enc_data.append((p, os.path.join(img_dir, img_file)))
    return enc_data

def _load_paths(data_dir, sort=True, subdir='*'):
    paths = []
    img_data_dir = data_dir / "image_data"
    #print("IMAGE DAGA DIR", img_data_dir, 'subdir', subdir)
    
    for p in tqdm(
        Path(img_data_dir).glob(f"{subdir}/*.json"),
        desc=f"loading dataset paths from {str(img_data_dir)}/{subdir}",
    ):
        paths.append(p)
    return sorted(paths)

class LazyLoader:
    def __init__(self, data_dir):
        self.paths = _load_paths(data_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = load_json(self.paths[idx])
        if data is None:
            return self[random.randint(0, len(self) - 1)]
        return data


class ImgCptDataset(Dataset):
    """
    Dataset which loads image caption data from MAGMA standard format and transforms them into tensors that can be input to the model.
    Images are expected to be stored in data_dir/images, image data in data_dir/image_data and each data item is a json file with format {"image_path": img_path, "captions": [caption1, caption2,...], "metadata":{...}}
    """

    def __init__(
        self, data_dir, tokenizer, transforms, subdir='*',seq_len=2048, load_data_in_memory=False
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.seq_len = seq_len
        self.load_data_in_memory = load_data_in_memory
        if self.load_data_in_memory:
            self.data = _bak_read_image_data(self.data_dir, subdir=subdir)
        else:
            self.data = LazyLoader(self.data_dir)
        print("LENGTH OF DATA", len(self.data))
    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx
    ) -> Tuple[TensorType["b", "c", "h", "w"], TensorType["b", "s"]]:
        img_data = self.data[idx]
        try:
            try:
                img_path = self.data_dir / img_data["image_path"]
            except KeyError as e:
                # if no image path is found, assume path is same as .json, but .jpg
                if not self.load_data_in_memory:
                    p = self.data.paths[idx]
                    img_path = (
                        self.data_dir
                        / "images"
                        / Path(p.parent).name
                        / Path(p.name).with_suffix(".jpg")
                    )
                else:
                    raise e
            img = Image.open(img_path).convert('RGB')
            #print("IMG", img)
            #print("IMG AGAIN", img.size)
            #print("transforms", self.transforms(img).shape)
            img_tensor = self.transforms(img).squeeze()
            caption = random.choice(img_data["captions"])
            #print("Captioninn", caption)
            caption_tensor = self.tokenizer.encode(
                caption,
                return_tensors="pt",
                max_length=self.seq_len,
                padding="longest",
                truncation=True,
            ).squeeze()
            return {'images':img_tensor, 'captions':caption_tensor}
        except (
            UnidentifiedImageError,
            OSError,
            DecompressionBombError,
            IndexError,
        ) as e:
            # return random index if image is corrupt
            print(f"Warning: Could not load image {str(img_path)}")
            return self[random.randint(0, len(self) - 1)]


def collate_fn(batch_data: List[Tuple[torch.Tensor, torch.Tensor]], seq_len=2048):

    all_images, all_captions = list(
        zip(*batch_data)
    )  # [(img1, caption1), (img2, caption2), ... ] -> [(img1, img2, ... ), (caption1, caption2, ... )]
    return torch.cat(all_images), torch.cat([i[:, :seq_len] for i in all_captions])

class ImgDataset(Dataset):
    """
    Dataset which loads images only. Returns the image and the filename
    """
    def __init__(self, data_dir, transforms, subdir='*', load_data_in_memory=False):
        self.data_dir = Path(data_dir)
        print("DATA DIR", data_dir, "SUBDIR", subdir)
        self.transforms = transforms
        self.load_data_in_memory = load_data_in_memory
        if self.load_data_in_memory:
            self.data = _read_image_data(self.data_dir, subdir=subdir)
        else:
            self.data = LazyLoader(self.data_dir) #TODO this won't ever work

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        try:
            try:
                img_path = self.data_dir / img_data["image_path"]
            except KeyError as e:
                # if no image path is found, assume path is same as .json, but .jpg
                if not self.load_data_in_memory:
                    p = self.data.paths[idx]
                    img_path = (
                        self.data_dir
                        / "images"
                        / Path(p.parent).name
                        / Path(p.name).with_suffix(".jpg")
                    )
                else:
                    raise e
            img = Image.open(img_path)
            img_tensor = self.transforms(img)
            
            return img_tensor, img_data["image_path"]
        except (
            UnidentifiedImageError,
            OSError,
            DecompressionBombError,
            IndexError,
        ) as e:
            # return random index if image is corrupt
            print(f"Warning: Could not load image {str(img_path)}")
            return None, img_path or img_data['image_path']
            #return self[random.randint(0, len(self) - 1)]



def _load_image_data(data_dir, encs_path, subdir='*'):
    jsonpaths = []
    paths = []
    img_data_dir = data_dir / "image_data"
    for p in tqdm(
        Path(img_data_dir).glob(f"{subdir}/*.json"),
        desc=f"loading dataset paths from {str(data_dir)}/{subdir}",
    ):
        jsonpaths.append(p)
    pbar = tqdm(
        jsonpaths,
        desc=f"loading dataset from {str(data_dir)}",
    )
    # read data with multiprocessing
    with Pool(cpu_count()) as pool:
        for img_data in pool.imap(load_json, pbar):
            if img_data is not None:
                if "0" not in img_data['metadata']:
                    continue
                img_id=img_data['metadata']['0']['image_id']
                img_file = img_data['image_path']
                img_path = os.path.join(data_dir, img_file)
                unique_path = '/'.join(img_file.split('/')[1:]) #"images/0/000000391895.jpg" -> "0/000000391895.jpg"
                enc_path = os.path.join(encs_path, unique_path+'.npy') #"vanilla_viz_encodings/clip_resnet_large/0/000000391895.jpg"
                paths.append((img_path, enc_path, img_id))
    return paths


class EncCptDataset(Dataset):
    #This is basically the same as ImgCaptionDataset but loads encodings rather than images
    #this is because we use this when we are not loading the image encoder (since we would be freezing it anyways)

    def __init__(self, data_dir, encs_folder,tokenizer, seq_len=2048, load_data_in_memory=False):
        '''
        encs_folder is either like vanilla_viz_encodings/clip_resnet_large or magma_viz_encodings/clip_resnet_large
        '''
        self.data_dir = Path(data_dir)
        self.encs_path = os.path.join(data_dir, encs_folder)
        self.tokenizer=tokenizer
        self.seq_len = seq_len
        self.load_data_in_memory = load_data_in_memory
        #print("image data", data_dir)
        if self.load_data_in_memory:
            self.data = _read_image_data(self.data_dir)
        else:
            self.data = LazyLoader(self.data_dir)
        #self.data = self.data[:10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        #image_data_path = self.data[idx]
        
        #TODO: filter these out earlier
        #assert "0" in img_data['metadata']
        #img_id=img_data['metadata']['0']['image_id']
        #print(image_data_path, img_id, img_file, img_data['captions'])
        #print("PATHS", enc_path, img_path)
        try:
            img_file = img_data['image_path']
            img_path = self.data_dir / img_file
        except KeyError as e:
            print(e, img_data)
        try:
            unique_path = '/'.join(img_file.split('/')[1:]) #"images/0/000000391895.jpg" -> "0/000000391895.jpg"
            enc_path = os.path.join(self.encs_path, unique_path+'.npy') #"vanilla_viz_encodings/clip_resnet_large/0/000000391895.jpg"
        except Exception as e:
            print("Can not create encoding path:", img_path, e)

        try:
            encs = torch.tensor(np.load(enc_path)).squeeze()
            caption = random.choice(img_data["captions"])
            caption_tensor = self.tokenizer.encode(
                caption,
                return_tensors="pt",
                max_length=self.seq_len,
                padding="longest",
                truncation=True,
            ).squeeze()
            #print("captino TENSOR", caption_tensor.shape)
            #if len(encs.shape)>1: #reverse channels and (h*w) axis
            #    encs = encs.transpose(1,0)
            #print('imgs shape in getitem', img_tensor.shape)
            torch.cuda.empty_cache()
            return {"images":encs, 'captions':caption_tensor}#.float() #if mixed precisin doesn't work, add .float() to both encs and img_tensor
        except (
            UnidentifiedImageError,
            OSError,
            DecompressionBombError,
            IndexError,
        ) as e:
            # return random index if image is corrupt
            print(f"Warning: Could not load image {str(enc_path)}",e)
            return self[random.randint(0, len(self) - 1)]

