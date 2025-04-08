#!/usr/bin/env python3

import cv2 as c
import numpy as np

def create_new_buffer(w: int, h: int, channels: int):
    return np.zeros((h, w, channels), dtype=np.uint8)

if __name__ == "__main__":
    image = c.imread("./cheekibreeki.jpg")

    print(
        """\
        1. Save as `inverse.jpg
        2. Preview inversed image
        3. Save as .bmp file
        4. Save as .pgm file\
        """
    )

    while True:
        user_chooice = int(input("Masukkan mode : "))

        grayscale = c.cvtColor(image, c.COLOR_BGR2GRAY)
        inverse_img = c.bitwise_not(grayscale)

        if user_chooice == 1:
            print(f" :: image saved as `inverse.jpg`.")
            _ = c.imwrite("inverse.jpg", inverse_img)
            continue

        if user_chooice == 2:
            print(f" :: showing images.")
            c.imshow("Inverse Image", inverse_img)
            _ = c.waitKey(0)
            continue

        tile = 32
        buffer = create_new_buffer(512, 512, 3)
        for i in range(0, 512, tile):
            for j in range(0, 512, tile):
                buffer[i:i+tile, j:j+tile, 0] = 0
                buffer[i:i+tile, j:j+tile, 1] = 0
                buffer[i:i+tile, j:j+tile, 2] = 255 if (i + j) % (tile*2) == 0 else 0


        file_name = "checker";

        if user_chooice == 3:
            print(f" :: Saving as {file_name}.bmp")
            _ = c.imwrite(f"{file_name}.bmp", buffer)
            continue

        if user_chooice == 4:
            print(f" :: Saving as {file_name}.pgm")
            gray_buffer = c.cvtColor(buffer, c.COLOR_BGR2GRAY)
            _ = c.imwrite(f"{file_name}.pgm", gray_buffer)
            continue

        if user_chooice >= 4:
            break
