import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", required=True, help="Input file to apply point parser to.")
parser.add_argument("-o", "--output", required=True, help="Output file. *.txt or any image file extention.")
configs = parser.parse_args()

image = cv2.imread(configs.file)

# Convert the image to binary
_, thresh = cv2.threshold(image[:, :, 2], 128, 255, 0)

# Find contours in the image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

centers = []

# For each contour, find the center and calculate the radius based on contour area
for contour in contours:
    moments = cv2.moments(contour)

    # Calculate center
    center_x = int(moments["m10"] / (moments["m00"] + 0.0001))
    center_y = int(moments["m01"] / (moments["m00"] + 0.0001))

    # Calculate radius based on contour area
    area = cv2.contourArea(contour)
    radius = (area / 3.14) ** 0.5  # Assuming circular shape

    if center_x + center_y:
        centers.append((center_x, center_y, radius))

if configs.output.endswith(".txt"):
    with open(configs.output, "w") as f:
        f.write("\n".join([str(point) for point in centers]))
    exit()

if configs.output.endswith(".csv"):
    with open(configs.output, "w") as f:
        f.write("x,y,r\n")
        f.write("\n".join([str(point)[1:-1] for point in centers]))


point_map_2d = np.zeros((image.shape))

for point in centers:
    radius = int(point[2])
    if radius >= 1:
        for x_shift in range(-radius, radius):
            for y_shift in range(-radius, radius):
                if x_shift == 0 and y_shift == 0:
                    continue

                if (point[1] + x_shift) < 0 or (point[1] + x_shift) >= image.shape[0]:
                    continue
                if (point[0] + y_shift) < 0 or (point[0] + y_shift) >= image.shape[1]:
                    continue
                
                point_map_2d[point[1] + x_shift][point[0] + y_shift][0] = 255

    point_map_2d[point[1]][point[0]][0] = 255


cv2.imwrite(configs.output, point_map_2d)