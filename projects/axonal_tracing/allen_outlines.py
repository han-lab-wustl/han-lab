from allensdk.api.queries.image_download_api import ImageDownloadApi
from allensdk.api.queries.svg_api import SvgApi
from allensdk.config.manifest import Manifest

import matplotlib.pyplot as plt
from skimage.io import imread
import pandas as pd
from pathlib import Path

import logging
import os
from base64 import b64encode

from IPython.display import HTML, display
%matplotlib inline

def verify_image(file_path, figsize=(18, 22)):
    image = imread(file_path)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    
    
def verify_svg(file_path, width_scale, height_scale):
    # we're using this function to display scaled svg in the rendered notebook.
    # we suggest that in your own work you use a tool such as inkscape or illustrator to view svg
    
    with open(file_path, 'rb') as svg_file:
        svg = svg_file.read()
    encoded_svg = b64encode(svg)
    decoded_svg = encoded_svg.decode('ascii')
    
    st = r'<img class="figure" src="data:image/svg+xml;base64,{}" width={}% height={}%></img>'.format(decoded_svg, width_scale, height_scale)
    display(HTML(st))

image_api = ImageDownloadApi()
svg_api = SvgApi()

#%%
output_dir = r'D:'
from allensdk.api.queries.svg_api import SvgApi
atlas_image_id=100960049
svg_api = SvgApi()
svg_api.download_svg(atlas_image_id, file_path=Path(output_dir) / f'{atlas_image_id}.svg')

verify_svg(Path(output_dir) / f'{atlas_image_id}.svg', 35, 35)
