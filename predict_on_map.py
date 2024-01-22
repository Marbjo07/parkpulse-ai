import os
import torch
import inspect
import argparse
import torch.nn as nn
from tqdm import trange
from torchvision import transforms
from PIL import Image, ImageEnhance
from torchvision.utils import save_image


def log(*args, level=1):
    if configs.verbose >= level: 
        print(args)

def get_parrent_dir(file_path):
    parrent_dir = file_path
    parrent_dir += "/" if parrent_dir[-1] != "/" else None
    parrent_dir = parrent_dir.split("/")
    parrent_dir = parrent_dir[:-2]
    parrent_dir = "/".join(parrent_dir)

    return parrent_dir

def save_tensor_as_img(x:torch.Tensor, file_path):
    x = x.permute(0, 2, 1)
    x = x.repeat(3, 1, 1)
    x[1, :, :] = 0
    x[2, :, :] = 0

    save_image(x, file_path)

def make_parking_lot_heatmap(input_file, output_file, image_size, kernel_size, stride, confidence_threshold, model_save_file):
    log("-"*50)
    log(f"{inspect.currentframe().f_code.co_name}() recived:")
    log(f"\tinput_file:           {input_file}")
    log(f"\toutput_file:          {output_file}")
    log(f"\timage_size:           {image_size}")
    log(f"\tkernel_size:          {kernel_size}")
    log(f"\tstride:               {stride}")
    log(f"\tconfidence_threshold: {confidence_threshold}")
    log(f"\tmodel_save_file:      {model_save_file}")
    log("-"*50)

    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))
        ])

    log("loading image...")
    image = Image.open(input_file).convert("RGB")

    log("\t", image)
    img = transform(image)
    del image
    log("\t", img.shape)

    if img.shape[1] != img.shape[2]:
        print("WARNING: Input image isn't a square. This could decrease the models performance. Make sure to sample images correctly from google maps. ")

    if img.shape[1] != image_size:
        print(f"WARNING: Image size isn't ({image_size}, {image_size}, 3). This could decrease the models performance. Make sure to sample images correctly from google maps. ")


    log("transforming image...")
    patches = img.unsqueeze(0).unfold(3, kernel_size, stride).unfold(2, kernel_size, stride)
    log("\t", patches.shape) # [B, C, nb_patches_h, nb_patches_w, kernel_size, kernel_size]
    del img

    log("\t", patches.shape)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    log("\t", patches.shape)



    activations = []

    log("running model on patches...")
    for line in trange(patches.size(1), leave=False):
        imgs = patches[0, line, :, :, :, :].to(device)
        with torch.no_grad():
            output = model(imgs)

        activations += [output]

    del patches

    log("finishing touches...")
    heatmap = torch.stack(activations)
    del activations

    heatmap = torch.threshold(heatmap, threshold=confidence_threshold, value=0)

    heatmap = heatmap.permute(2, 1, 0).unsqueeze(0)
    log("\t", heatmap.shape)

    log("saving heatmap...")
    save_tensor_as_img(heatmap[0], output_file)
    del heatmap

    log("Heatmap is now applied.")


def apply_heatmap_to_img(input_file, output_file, heatmap_file):
    log("-"*50)
    log(f"{inspect.currentframe().f_code.co_name}() recived:")
    log(f"\tinput_file:   {input_file}")
    log(f"\toutput_file:  {output_file}")
    log(f"\theatmap_file: {heatmap_file}")
    log("-"*50)

    log("loading input image...")
    image = Image.open(input_file)
 
    log("loading heatmap...")
    overlay = Image.open(heatmap_file)

    log("resizing images...")
    image = image.resize((image_size, image_size))
    overlay = overlay.resize((image_size, image_size))

    log("converting image types...")
    image = image.convert("RGBA")
    overlay = overlay.convert("RGBA")

    log("overlaying images...")

    new_img = Image.blend(image, overlay, 0.6)

    log("increasing brightness...")
    enhancer = ImageEnhance.Brightness(new_img)
    # to reduce brightness by 50%, use factor 0.5
    new_img = enhancer.enhance(1.5)

    log("saving results...")
    new_img.save(output_file,"PNG")
    log("Heatmap is now applied to input image.")



parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input_file", required=True, help="Input file or folder to apply model on. If a folder every image will be processed")
parser.add_argument("-s", "--stride", required=False, default=64, help="Stride of model kernel. A higher value increases accuracy but also increases compute time and noise (compute complexity is squared).", type=int)
parser.add_argument("-c", "--confidence", required=False, default=0.9, help="Threshold for clipping the heatmap. Higher values reduce noise but may sacrifice accuracy.", type=float)
parser.add_argument("-v", "--verbose", required=False, default=1, help="Verbosity", type=int)
configs = parser.parse_args()

confidence_threshold = configs.confidence

input_file = configs.input_file


image_size = 2560
kernel_size = 128

custom_stride = configs.stride != kernel_size // 2

if configs.stride >= kernel_size:
    print(f"WARNING: Setting stride to greater than or equal to {kernel_size} leads to bad model performance, as it will miss parts of the image. Recommended stride: {kernel_size // 2}:")


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


output_dir = get_parrent_dir(input_file)
if os.path.isdir(input_file):
    input_file += "/" if input_file[-1] != "/" else None
    output_dir += "/" if output_dir[-1] != "/" else None
    
    print(output_dir + 'output', output_dir + 'heatmaps')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + 'output', exist_ok=True)
    os.makedirs(output_dir + 'heatmaps', exist_ok=True)

    files = (file for file in os.listdir(input_file) 
             if os.path.isfile(os.path.join(input_file, file)))
    input_files = []
    output_files = []
    heatmap_files = []
    for file in files:
        file_name = ".".join(file.split('.')[:-1])

        if custom_stride:
            file_name += f'_stride_{configs.stride}'

        input_files += [f'{input_file}{file}']
        output_files += [f'{output_dir}output/{file_name}.png']
        heatmap_files += [f'{output_dir}heatmaps/{file_name}.png']

else:
    
    print(output_dir)
    custom_stride_info = f'_stride_{configs.stride}' if custom_stride else ''

    output_files = [f'{output_dir}/output{custom_stride_info}.png']
    heatmap_files = [f'{output_dir}/heatmap{custom_stride_info}.png']

assert len(input_files) == len(output_files)
assert len(output_files) == len(heatmap_files)

for input_file, output_file, heatmap_file in zip(input_files, output_files, heatmap_files):
    print(output_file, heatmap_file)

    make_parking_lot_heatmap(input_file, 
                         heatmap_file, 
                         image_size, 
                         kernel_size, 
                         configs.stride,
                         confidence_threshold,
                         model_save_file)


    apply_heatmap_to_img(input_file, 
                         output_file, 
                         heatmap_file)
    



# python -m predict_on_map -f ./images/example_input.png
