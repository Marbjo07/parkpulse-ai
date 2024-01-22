from PIL import Image
import os
from tqdm import trange

# Assuming your images are in the ./raw_images/ directory
directory = './raw_images/'
output_dir = './tiles/'

# Size of each individual image
image_size = (256, 256)  # Adjust this based on the actual size of your images

base_y = 77057
base_x = 144171


# number of samples from google maps
raw_images_n_rows = 100
raw_images_n_cols = 100

# Size of the final image
n_rows = 10
n_cols = 10
final_size = (n_rows * image_size[0], n_cols * image_size[1])


# Iterate through the images and paste them into the final image
for image_x in trange(raw_images_n_rows // n_rows):
    for image_y in trange(raw_images_n_cols // n_cols):
        
        # Create a new blank image with the final size
        final_image = Image.new('RGB', final_size)

        for x in range(n_rows):
            for y in range(n_cols):
                image_path = f'{directory}img_{base_x + x + image_x * n_rows}_{base_y + y + image_y * n_cols}.png'

                # Check if the image file exists
                if os.path.exists(image_path):
                    # Open the image file
                    image = Image.open(image_path)

                    # Calculate the position to paste the image
                    paste_position = (x * image_size[0], y * image_size[1])

                    # Paste the image into the final image
                    final_image.paste(image, paste_position)

        # Save the final image
        final_image.save(f'{output_dir}img_{image_x}_{image_y}.png')