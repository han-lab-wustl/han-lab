import numpy as np, os
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
import tifffile as tif
# input data (can be any numpy compatible array: numpy, dask, cupy, etc.)
src = r'X:\rna_fish_alignment_zstacks\240702\e217\240702_ZD_001_001'
tile_arrays = [tif.imread(os.path.join(src,xx)) for xx in os.listdir(src) if 'registered.tif' in xx]

# indicate the tile offsets and spacing
tile_translations = [
    {"z": 2.5, "y": 0, "x": 0},
    {"z": 2.5, "y": 100, "x": -100},
    {"z": 2.5, "y": 200, "x": -200},
    {"z": 2.5, "y": 0, "x": 0},
    {"z": 2.5, "y": 100, "x": -100},
    {"z": 2.5, "y": 200, "x": -200},
    {"z": 2.5, "y": 0, "x": 0},
    {"z": 2.5, "y": 100, "x": -100},
    {"z": 2.5, "y": 200, "x": -200},
]
spacing = {"z": 5, "y": 0.5, "x": 0.5}

channels = ["GFP"]

# build input for stitching
msims = []
for tile_array, tile_translation in zip(tile_arrays, tile_translations):
    sim = si_utils.get_sim_from_array(
        tile_array,
        dims=["c", "z", "y", "x"],
        scale=spacing,
        translation=tile_translation,
        transform_key="stage_metadata",
        c_coords=channels,
    )
    msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))

# plot the tile configuration
# from multiview_stitcher import vis_utils
# fig, ax = vis_utils.plot_positions(msims, transform_key='stage_metadata', use_positional_colors=False)