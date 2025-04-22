# 512x512 abu abu

import numpy as np
import cv2

if __name__ == "__main__":
    canvas = 512
    color = 127
    img = np.zeros((canvas, canvas, 1), dtype=np.uint8)
    img.fill(color)
    cv2.imwrite("canvas.png", img)
    pass
