import os
import torch
import inspect
import argparse
import torch.nn as nn
from tqdm import trange
from torchvision import transforms
from PIL import Image, ImageEnhance
from torchvision.utils import save_image


def save_tensor_as_img(x:torch.Tensor, file_path):
    x = x.permute(0, 2, 1)
    x = x.repeat(3, 1, 1)
    x[1, :, :] = 0
    x[2, :, :] = 0

    save_image(x, file_path)

def make_parking_lot_heatmap(input_file, output_file, image_size, kernel_size, stride, model_save_file):
    print("-"*50)
    print(f"{inspect.currentframe().f_code.co_name}() recived:")
    print(f"\tinput_file:      {input_file}")
    print(f"\toutput_file:     {output_file}")
    print(f"\timage_size:      {image_size}")
    print(f"\tkernel_size:     {kernel_size}")
    print(f"\tstride:          {stride}")
    print(f"\tmodel_save_file: {model_save_file}")
    print("-"*50)

    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))
        ])

    print("loading image...")
    image = Image.open(input_file).convert("RGB")

    print("\t", image)
    img = transform(image)
    del image
    print("\t", img.shape)

    if img.shape[1] != img.shape[2]:
        print("WARNING: Input image isn't a square. This could decrease the models performance. Make sure to sample images correctly from google maps. ")

    if img.shape[1] != image_size:
        print(f"WARNING: Image size isn't ({image_size}, {image_size}, 3). This could decrease the models performance. Make sure to sample images correctly from google maps. ")


    print("transforming image...")
    patches = img.unsqueeze(0).unfold(3, kernel_size, stride).unfold(2, kernel_size, stride)
    print("\t", patches.shape) # [B, C, nb_patches_h, nb_patches_w, kernel_size, kernel_size]
    del img

    print("\t", patches.shape)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    print("\t", patches.shape)



    activations = []

    print("running model on patches...")
    for line in trange(patches.size(1)):
        imgs = patches[0, line, :, :, :, :].to(device)
        with torch.no_grad():
            output = model(imgs)

        activations += [output]

    del patches

    print("finishing touches...")
    heatmap = torch.stack(activations)
    del activations

    heatmap = torch.threshold(heatmap, threshold=0.9, value=0)

    heatmap = heatmap.permute(2, 1, 0).unsqueeze(0)
    print("\t", heatmap.shape)

    print("saving heatmap...")
    save_tensor_as_img(heatmap[0], output_file)
    del heatmap

    print("Heatmap is now applied.")


def apply_heatmap_to_img(input_file, output_file, heatmap_file):
    print("-"*50)
    print(f"{inspect.currentframe().f_code.co_name}() recived:")
    print(f"\tinput_file:   {input_file}")
    print(f"\toutput_file:  {output_file}")
    print(f"\theatmap_file: {heatmap_file}")
    print("-"*50)

    print("loading input image...")
    image = Image.open(input_file)
 
    print("loading heatmap...")
    overlay = Image.open(heatmap_file)

    print("resizing images...")
    image = image.resize((image_size, image_size))
    overlay = overlay.resize((image_size, image_size))

    print("converting image types...")
    image = image.convert("RGBA")
    overlay = overlay.convert("RGBA")

    print("overlaying images...")

    new_img = Image.blend(image, overlay, 0.6)

    print("increasing brightness...")
    enhancer = ImageEnhance.Brightness(new_img)
    # to reduce brightness by 50%, use factor 0.5
    new_img = enhancer.enhance(1.5)

    print("saving results...")
    new_img.save(output_file,"PNG")
    print("Heatmap is now applied to input image.")



parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input_file", required=True, help="input file to apply model on")
parser.add_argument("-s", "--stride", required=False, default=32, help="stride of model kernel. A higher value increases accuracy but also increases compute time (compute complexity is squared).", type=int)
configs = parser.parse_args()
input_file = configs.input_file

os.makedirs("./images", exist_ok=True)

output_file = f'./images/output_stride_{configs.stride}.png'
heatmap_file = "./images/heatmap.png"

image_size = 2560
kernel_size = 128

if configs.stride >= kernel_size:
    print(f"WARNING: Setting stride to greater than or equal to {kernel_size} leads to bad model performance, as it will miss parts of the image.")


model_save_file = "./model/pretrained_model.pth"

print("loading model...")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = nn.Sequential(torch.load("./model/part1.pth"), torch.load("./model/part2.pth"))
model.eval()

model = model.to(device)

if torch.cuda.is_available():
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

make_parking_lot_heatmap(input_file, 
                 heatmap_file, 
                 image_size, 
                 kernel_size, 
                 configs.stride,
                 model_save_file)


apply_heatmap_to_img(input_file, 
                     output_file, 
                     heatmap_file)


# python -m predict_on_map -f ./images/example_input.png
