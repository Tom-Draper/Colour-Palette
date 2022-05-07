import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans


def colour_clusters(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    colours = kmeans.cluster_centers_
    return colours.astype(int)

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

def colour_code(rgb):
    return ('#%02x%02x%02x' % rgb).upper()

def draw_colour_code(draw, rgb, N, img_height, block_size):
    font = ImageFont.truetype("Lato/Lato-Bold.ttf", size=block_size//10)
    hex_colour = colour_code(rgb)
    draw.text((block_size//2 - (block_size//4.7) + N * block_size, img_height + (block_size//2) - (block_size//20)), 
              hex_colour, 
              'black', 
              font=font)

def run(path, n_clusters=5, output='colours'):
    print('Running...')
    try:
        arr = load_image(path)
    except AttributeError as e:
       print('Error: Image file not found.')
       return
    
    print('Image size:', arr.shape)
    X = arr.reshape(-1, arr.shape[-1])  # Create single vector of all rgb values

    colours = colour_clusters(X, n_clusters)
    print('Colours used:', [tuple(colour) for colour in colours])
    
    colour_blocks = build_colour_blocks(arr.shape[1], colours)
    img = Image.fromarray(np.concatenate((arr, colour_blocks), axis=0))
    
    draw = ImageDraw.Draw(img)
    for i, rgb in enumerate(colours):
        draw_colour_code(draw, tuple(rgb), i, arr.shape[0], arr.shape[1]//n_clusters)
    
    img.save(output + '.png')

if __name__ == '__main__':
    path = None
    n_clusters = 5
    output = 'colours'
    for i, arg in enumerate(sys.argv[1:]):
        if i != len(sys.argv)-1:
            if arg == '-f':
                path = sys.argv[i+2]
            elif arg == '-c':
                n_clusters = sys.argv[i+2]
            elif arg == '-o':
                output = sys.argv[i+2]

    if path == None:
        print('Error: Path to image file required with cmdline flag -f <path>')
    else:
        run(path=path, n_clusters=n_clusters, output=output)