import cv2
import numpy as np
from PIL import Image

def points_to_polygon(points, image_size):
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    if len(points) >= 3:
        polygon = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 1)

    return mask
    
def draw_lanes(image, lanes, color=(0, 255, 0), thickness=2):
    image = np.array(image.copy())
    height, width = image.shape[:2]

    for lane in lanes:
        if 'points' in lane:
            points = lane['points']
            for i in range(len(points) - 1):
                pt1 = tuple(map(int, points[i]))
                pt2 = tuple(map(int, points[i + 1]))
                cv2.line(image, pt1, pt2, color, thickness)

        elif 'params' in lane:
            a1, a2, a3, a4, a5 = lane['params']
            offset = lane['offset']
            ys_max = lane.get('ys_max', height)  # Fallback to image height

            ys = np.linspace(offset, ys_max, num=50)
            ys_norm = (ys - offset) / (ys_max - offset + 1e-8)

            xs = []
            for y_norm in ys_norm:
                # ✅ Multiply by width to denormalize x
                x = (a1 * (y_norm ** 4) + a2 * (y_norm ** 3) + a3 * (y_norm ** 2) + a4 * y_norm + a5) * width
                xs.append(x)

            # Draw lines between consecutive points
            for i in range(len(xs) - 1):
                pt1 = (int(xs[i]), int(ys[i]))
                pt2 = (int(xs[i + 1]), int(ys[i + 1]))

                # ✅ Add check to keep points inside image boundaries
                if (0 <= pt1[0] < width and 0 <= pt2[0] < width and
                    0 <= pt1[1] < height and 0 <= pt2[1] < height):
                    cv2.line(image, pt1, pt2, color, thickness)

    return image

