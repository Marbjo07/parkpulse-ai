import cv2
from PIL import Image
import os
from random import random


def save_patch(image_cv2, source_image, x, y, car, scale_ratio, dir_path, train_split_ratio):
    
    half_image_size = image_size // 2
    x_scaled = int(x * (1 / scale_ratio))
    y_scaled = int(y * (1 / scale_ratio))

    print('x = %d, y = %d'%(x, y))

    color = (255, 0, 0) if car else (0, 0, 255)
    label = 'car' if car else 'neg'
    train_split = 'test' if random() < train_split_ratio else 'train'
    

    top_left = (x - half_image_size, y - half_image_size)
    bottom_right = (x + half_image_size, y + half_image_size)

    cv2.rectangle(image_cv2, top_left, bottom_right, color, 3)
    cv2.imshow("window", image_cv2)

    cropped = source_image.crop((x_scaled - half_image_size, y_scaled - half_image_size, x_scaled + half_image_size, y_scaled + half_image_size))
    cropped.save(f'{dir_path}/{train_split}/{label}_crop_{tile_x}_{tile_y}_{x_scaled}_{y_scaled}.png', 'PNG')


if __name__ == "__main__":

    data_root = "./tiles/"
    scale_ratio = 0.5
    train_split_ratio = 0.2
    image_size = 128
    
    start_x, end_x = 0, 9
    start_y, end_y = 0, 9

    os.mkdir('./dataset', exist_ok=True)

    for tile_x in range(start_x, end_x + 1):
        for tile_y in range(start_y, end_y + 1):

            file_name = f"img_{tile_x}_{tile_y}.png"

            dir_path = f"./dataset/{file_name[:-4]}"
            os.mkdir(dir_path, exist_ok=True)
            os.mkdir(dir_path + '/train', exist_ok=True)
            os.mkdir(dir_path + '/test', exist_ok=True)

            image_cv2 = cv2.imread(data_root + file_name)
            height, width, _ = image_cv2.shape
            image_cv2 = cv2.resize(image_cv2, (int(width * scale_ratio), int(height * scale_ratio))) 

            source_image = Image.open(data_root + file_name)


            cv2.imshow("window", image_cv2)

            def onMouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    save_patch(image_cv2, source_image, x, y, True, scale_ratio, dir_path, train_split_ratio)
                elif event == cv2.EVENT_RBUTTONDOWN:
                    save_patch(image_cv2, source_image, x, y, False, scale_ratio, dir_path, train_split_ratio)

            cv2.setMouseCallback('window', onMouse)

            cv2.waitKey(0)