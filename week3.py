import numpy as n
import cv2 as cv

def translate_image(image, output, tx, ty):
    height, width = image.shape[:2]
    translation_mat = n.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=n.float32)
    new_image = cv.warpPerspective(image, translation_mat, (width, height))
    cv.imwrite(output, cv.cvtColor(new_image, cv.COLOR_RGB2BGR))
    pass

def rotate_image_euclidean(image, output, angle, anchor=None):
    """
    Rotate an image using Euclidean transformation (rotation + translation).

    Args:
        image: Input image
        output: Output file path
        angle: Rotation angle in degrees
        anchor: Center of rotation (x, y). If None, image center is used.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    if anchor is not None:
        center = anchor

    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv.warpAffine(image, rot_mat, (width, height))
    cv.imwrite(output, cv.cvtColor(new_image, cv.COLOR_RGB2BGR))

def rotate_image_similarity(image, output, angle, scale=1.0, anchor=None):
    """
    Rotate and scale an image using similarity transformation (rotation + translation + uniform scaling).

    Args:
        image: Input image
        output: Output file path
        angle: Rotation angle in degrees
        scale: Scaling factor (uniform in all directions)
        anchor: Center of rotation (x, y). If None, image center is used.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    if anchor is not None:
        center = anchor

    rot_mat = cv.getRotationMatrix2D(center, angle, scale)
    new_image = cv.warpAffine(image, rot_mat, (width, height))
    cv.imwrite(output, cv.cvtColor(new_image, cv.COLOR_RGB2BGR))

def scale_image(image, output, sx, sy):
    height, width = image.shape[:2]
    scale_mat = n.array([
        [sx, 0 , 0],
        [0 , sy, 0],
        [0 , 0 , 1]
    ], dtype=n.float32)
    new_image = cv.warpPerspective(image, scale_mat, (width, height))
    cv.imwrite(output, cv.cvtColor(new_image, cv.COLOR_RGB2BGR))
    pass

def affine_transform_image(image, output):
    height, width = image.shape[:2]
    src_points = n.array([
        [50, 50],
        [200, 50],
        [50, 200]
    ], dtype=n.float32)
    dst_points = n.array([
        [10, 100],
        [200, 50],
        [100, 250]
    ], dtype=n.float32)
    affine_matrix = cv.getAffineTransform(src_points, dst_points)
    new_image = cv.warpAffine(image, affine_matrix, (width, height))
    cv.imwrite(output, cv.cvtColor(new_image, cv.COLOR_RGB2BGR))
    pass

def perspective_transform_image(image, output):
    height, width = image.shape[:2]
    src_points = n.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ], dtype=n.float32)
    dst_points = n.array([
        [50, 50],
        [width - 90, 50],
        [50, height - 50],
        [width - 50, height - 50]
    ], dtype=n.float32)
    perspective_matrix = cv.getPerspectiveTransform(src_points, dst_points)
    new_image = cv.warpPerspective(image, perspective_matrix, (width, height))
    cv.imwrite(output, cv.cvtColor(new_image, cv.COLOR_RGB2BGR))
    pass

def transpose_image(image, output, swap):
    # original shape : (height, width, channels)
    # original shape : (0     , 1    , 2       )
    # the `args 1, 0, 2` because : (width, height, channels)
    # new_image = n.transpose(image, (1, 0, 2))

    new_image = n.transpose(image, (swap[0], swap[1], swap[2]))
    cv.imwrite(output, cv.cvtColor(new_image, cv.COLOR_RGB2BGR))
    pass

def create_histogram(image, output):
    canvas_height = 300
    canvas_width = 256
    hist_img = n.zeros((canvas_height, canvas_width, 3), dtype=n.uint8)
    colors = ('b', 'g', 'r')
    channel_colors = {
        'b': (255, 0, 0),
        'g': (0, 255, 0),
        'r': (0, 0, 255)
    }

    for i, c in enumerate(colors):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        hist = cv.normalize(hist, hist, 0, canvas_height, cv.NORM_MINMAX)
        for x in range(1, 256):
            y1 = canvas_height - int(hist[x - 1][0])
            y2 = canvas_height - int(hist[x][0])
            cv.line(hist_img, (x - 1, y1), (x, y2), channel_colors[c], 1)
    cv.imwrite(output, hist_img)
    pass

def change_colorspace(image, output):
    grayscale_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite(output[0], grayscale_img)

    hsv_img = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    h_chan, s_chan, v_chan = cv.split(hsv_img)
    cv.imwrite(output[1], h_chan)
    cv.imwrite(output[2], s_chan)
    cv.imwrite(output[3], v_chan)
    pass

if __name__ == "__main__":
    image = cv.imread('image.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    translate_image(image, 'image-translated.png', 20, 20)
    rotate_image_euclidean(image, 'image-rotated-euclidean.png', 45)
    rotate_image_similarity(image, 'image-rotated-similarity.png', 50)
    scale_image(image, 'image-scaled.png', 2, 2)
    affine_transform_image(image, 'image-affinet.png')
    perspective_transform_image(image, 'image-perspectivet.png')
    transpose_image(image, 'image-transposed.png', (1, 0, 2))
    create_histogram(image, 'histogram.png')
    change_colorspace(image, [
        "image-grayscale.png",
        "image-hsv-h.png",
        "image-hsv-s.png",
        "image-hsv-v.png"
    ])

# -- NOT USED -- #
def traditional_project_point(coords: list, f: int):
    if coords[2] == 0:
        return None, None
    u = (coords[0] * f) / coords[2]
    v = (coords[1] * f) / coords[2]
    return u, v
