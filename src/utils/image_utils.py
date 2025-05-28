import numpy as np
from matplotlib import pyplot as plt
from seaborn.external.husl import hex_to_rgb

hex_monk_scale = [
    '#f6ede4',
    '#f3e7db',
    '#f7ead0',
    '#eadaba',
    '#d7bd96',
    '#a07e56',
    '#825c43',
    '#604134',
    '#3a312a',
    '#292420'
]

monk_scale = np.array([hex_to_rgb(h) for h in hex_monk_scale])

def get_median_color(image, mask=None):
    pixels = image.reshape(-1, 3)[~mask.reshape(-1).astype(bool)] if mask is not None else image.reshape(-1, 3)

    median_color = np.median(pixels, axis=0).astype(int)

    return median_color

def display_color(color):
    image = np.full((100, 100, 3), color, dtype=np.uint8)

    fig, ax = plt.subplots()

    ax.imshow(image)
    ax.set_title(f"Color: {color}")
    ax.axis('off')

    return fig

def hex_to_rgb(hex_color):
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]

    hex_value = int(hex_color, 16)

    red = (hex_value >> 16) & 0xFF
    green = (hex_value >> 8) & 0xFF
    blue = hex_value & 0xFF

    return np.array([red, green, blue])
