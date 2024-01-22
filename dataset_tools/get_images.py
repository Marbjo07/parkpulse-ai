import shutil

import requests
from tqdm import trange
import os


base_y = 77057
base_x = 144171

n_rows = 100
n_cols = 100

for x in trange(n_rows):
    for y in trange(n_cols, leave=False):
        if not os.path.exists(f'./raw_images/img_{base_x + x}_{base_y + y}.png'):
            url = f'https://khms3.google.com/kh/v=967?x={base_x + x}&y={base_y + y}&z=18'
            response = requests.get(url, stream=True)
            with open(f'./raw_images/img_{base_x + x}_{base_y + y}.png', 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response