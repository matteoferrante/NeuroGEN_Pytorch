## file to manage VQGAN, DALLE and other pretrained models


import sys
sys.path.append("../taming_transformers")

# also disable grad to save memory
import torch
from torchvision import datasets, transforms
from imutils import paths
import PIL
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import numpy as np
import yaml
import torch
from omegaconf import OmegaConf
from taming_transformers.taming.models.vqgan import VQModel, GumbelVQ
from dall_e import map_pixels, unmap_pixels, load_model



def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def reconstruct_with_vqgan(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    xrec = model.decode(z)
    return xrec

def preprocess_dalle(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    #if s < target_image_size:
    #    raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle: 
        img = map_pixels(img)
    return img


def preprocess_vqgan(x):
    return torch.nn.functional.interpolate(x,320,mode="bilinear")

def preprocess_dalle(x,map_dalle=True):
    img = torch.nn.functional.interpolate(x,320,mode="bilinear")
    if map_dalle: 
        img = map_pixels(img)
    return img


def init_vqgan(config_path,model_path):
    config = load_config(config_path, display=False)
    model = load_vqgan(config, ckpt_path=model_path)
    return model

def reconstruct_with_dalle(x, encoder, decoder, do_preprocess=False):
  # takes in tensor (or optionally, a PIL image) and returns a PIL image
    if do_preprocess:
        x = preprocess(x)
    z_logits = encoder(x)
    z = torch.argmax(z_logits, axis=1)

    print(f"DALL-E: latent shape: {z.shape}")
    z = F.one_hot(z, num_classes=encoder.vocab_size).permute(0, 3, 1, 2).float()

    x_stats = decoder(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    #x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

    return x_rec
