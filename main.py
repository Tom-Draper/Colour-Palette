from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import png


def colour_clusters(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    colours = kmeans.cluster_centers_
    for colour in colours:
        print(colour)
    return colours

def load_image(path):
    im = Image.open(path)
    arr = np.array(im)
    return arr

def build_colour_blocks(img_width, colours):
    block_size = img_width // len(colours)
    remaining = img_width - (block_size * len(colours))
    colour_blocks = np.full((block_size, block_size, 3), colours[0])
    
    for i, colour in enumerate(colours[1:]):
        if i == len(colours)-2:
            block = np.full((block_size, block_size+remaining, 3), colour)
        else:
            block = np.full((block_size, block_size, 3), colour)
        colour_blocks = np.concatenate((colour_blocks, block), axis=1)
    colour_blocks = np.array(colour_blocks, dtype=np.uint8)
    
    return colour_blocks

def run():
    arr = load_image('wave.jpg')
    print(arr.shape)
    X = arr.reshape(-1, arr.shape[-1])  # Create single vector of all rgb values

    colours = colour_clusters(X)
    
    colour_blocks = build_colour_blocks(arr.shape[1], colours)
    img = Image.fromarray(np.concatenate((arr, colour_blocks), axis=0))
    
    draw = ImageDraw.Draw(img)
    draw.text((100, 100), 'hidsgfsdagdsagfdsafdfasdf', 'black')
    
    img.save('colours.png')

if __name__ == '__main__':
    run()