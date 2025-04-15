import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#!git clone https://github.com/LTH14/fractalgen.git

#os.chdir('..')
#os.environ['PYTHONPATH'] = '/env/python:/content/fractalgen'
#!pip install timm==0.9.12
import torch
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

def compare_images(image1_path, image2_path):

    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    equal_pixels = np.all(arr1 == arr2, axis=-1)
    percentage = (np.sum(equal_pixels) * 1.0 / equal_pixels.size) * 100
    
    return percentage


print(compare_images("samples1.png", "samples2.png"))
model_type = "fractalmar_in64" #@param ["fractalar_in64", "fractalmar_in64"]
if model_type == "fractalar_in64":
  download.download_pretrained_fractalar_in64(overwrite=False)
  num_conds=1
elif model_type == "fractalmar_in64":
  download.download_pretrained_fractalmar_in64(overwrite=False)
  num_conds=5
else:
  raise NotImplementedError
model = fractalgen.__dict__[model_type](
    guiding_pixel=False,
    num_conds=num_conds,
).to(device)
state_dict = torch.load("pretrained_models/{}/checkpoint-last.pth".format(model_type))["model"]
model.load_state_dict(state_dict)
model.eval() # important!

# Set user inputs
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
np.random.seed(seed)
class_labels = 207, 360 #@param {type:"raw"}
num_iter_list = 64, 16 #@param {type:"raw"}
cfg_scale = 5 #@param {type:"slider", min:1, max:10, step:0.5}
cfg_schedule = "constant" #@param ["linear", "constant"]
temperature = 1.02 #@param {type:"slider", min:0.9, max:1.2, step:0.01}
filter_threshold = 1e-4
samples_per_row = 1 #@param {type:"number"}

label_gen = torch.Tensor(class_labels).long().to(device)
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
pix_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1)
pix_std = torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1)
sampled_images = sampled_images * pix_std + pix_mean
sampled_images = torch.nn.functional.interpolate(sampled_images, scale_factor=4, mode="nearest")
sampled_images = sampled_images.detach().cpu()

# Save & display images
save_image(sampled_images, "samples3.png", nrow=int(samples_per_row), normalize=True, value_range=(0, 1))
#samples = Image.open("samples3.png")

#from IPython.display import display, clear_output
#clear_output(wait=True)
#display(samples)

print(compare_images("samples1.png", "samples3.png"))

model_type = "fractalmar_large_in256" #@param ["fractalmar_base_in256", "fractalmar_large_in256", "fractalmar_huge_in256"]
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

# Set user inputs
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
np.random.seed(seed)
num_iter_list = 64, 16, 16 #@param {type:"raw"}
cfg_scale = 10 #@param {type:"slider", min:1, max:20, step:0.5}
cfg_schedule = "constant" #@param ["linear", "constant"]
temperature = 1.1 #@param {type:"slider", min:0.9, max:1.2, step:0.01}
filter_threshold = 1e-3
class_labels = 207, 360 #@param {type:"raw"}
samples_per_row = 1 #@param {type:"number"}

label_gen = torch.Tensor(class_labels).long().to(device)
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
pix_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1)
pix_std = torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1)
sampled_images = sampled_images * pix_std + pix_mean
sampled_images = sampled_images.detach().cpu()

# Save & display images
save_image(sampled_images, "samples4.png", nrow=int(samples_per_row), normalize=True, value_range=(0, 1))
#samples1 = Image.open("samples4.png")

#from IPython.display import display, clear_output
#clear_output(wait=True)
#display(samples1)


print(compare_images("samples2.png", "samples4.png"))