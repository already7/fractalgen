import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#!git clone https://github.com/LTH14/fractalgen.git

#os.chdir('..')
#os.environ['PYTHONPATH'] = '/env/python:/content/fractalgen'
#!pip install timm==0.9.12
import torch
import random 
import os
import numpy as np
from models import fractalgen
from torchvision.utils import save_image
from util import download
from PIL import Image
#from IPython.display import display
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
#device = "cpu"

import numpy as np

def seed_everything(seed: int) -> None:

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def compare_images(image1_path, image2_path):

    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    equal_pixels = np.all(arr1 == arr2, axis=-1)
    print("Количество совпавших пикселей:", np.sum(equal_pixels), "Количество пикселей:", equal_pixels.size)
    diff = np.abs(arr1.astype(np.int16) - arr2.astype(np.int16))
    total_diff = np.sum(diff)
    print("Суммарная разница между пикселями: ", total_diff)


compare_images('sdocker.png', 'samples.png')

seed = 0 #@param {type:"number"}
seed_everything(seed)
torch.backends.cudnn.deterministic = True

model_type = "fractalmar_base_in256" #@param ["fractalmar_base_in256", "fractalmar_large_in256", "fractalmar_huge_in256"]
num_conds = 5
if model_type == "fractalmar_base_in256":
  download.download_pretrained_fractalmar_base_in256(overwrite=False)
elif model_type == "fractalmar_large_in256":
  download.download_pretrained_fractalmar_large_in256(overwrite=False)
elif model_type == "fractalmar_huge_in256":
  download.download_pretrained_fractalmar_huge_in256(overwrite=False)
else:
  raise NotImplementedError
model = fractalgen.__dict__[model_type](
    guiding_pixel=True,
    num_conds=num_conds
).to(device)
state_dict = torch.load("pretrained_models/{}/checkpoint-last.pth".format(model_type))["model"]
model.load_state_dict(state_dict)
model.eval() # important!


num_iter_list = 64, 16, 16 #@param {type:"raw"}
cfg_scale = 10 #@param {type:"slider", min:1, max:20, step:0.5}
cfg_schedule = "constant" #@param ["linear", "constant"]
temperature = 1.1 #@param {type:"slider", min:0.9, max:1.2, step:0.01}
filter_threshold = 1e-3
class_labels = 207, 360 #@param {type:"raw"}
samples_per_row = 1 #@param {type:"number"}

label_gen = torch.Tensor(class_labels).long().cuda()
class_embedding = model.class_emb(label_gen)
if not cfg_scale == 1.0:
  class_embedding = torch.cat([class_embedding, model.fake_latent.repeat(label_gen.size(0), 1)], dim=0)

with torch.no_grad():
  with torch.cuda.amp.autocast():
    sampled_images = model.sample(
      cond_list=[class_embedding for _ in range(num_conds)],
      num_iter_list=num_iter_list,
      cfg=cfg_scale, cfg_schedule=cfg_schedule,
      temperature=temperature,
      filter_threshold=filter_threshold,
      fractal_level=0,
      visualize=False)

# Denormalize images.
pix_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(1, -1, 1, 1)
pix_std = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(1, -1, 1, 1)
sampled_images = sampled_images * pix_std + pix_mean
sampled_images = sampled_images.detach().cpu()

# Save & display images
save_image(sampled_images, "samples.png", nrow=int(samples_per_row), normalize=True, value_range=(0, 1))

compare_images('sdocker.png', 'samples.png')

input()