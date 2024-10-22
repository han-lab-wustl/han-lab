import SimpleITK as sitk, os
import numpy as np
import ffmpeg

import ffmpeg,re
import numpy as np
import os

# Function to extract the integer after '241018_E246_v2'
def extract_integer_from_basename(path,basename):
    # Extract basename part from the path
    filename = path.split("\\")[-1]
    
    # Search for the integer after the given basename pattern
    regex = rf'{re.escape(basename)}-(\d+)'
    match = re.search(regex, filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # Return infinity if no match found to push such files to the end


def vidwrite(fn, images, framerate=31.25*2, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    
    if not isinstance(fn, str) or not fn:
        raise ValueError("Output filename must be a non-empty string.")
    
    n, height, width = images.shape

    try:
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', r=framerate, pix_fmt='gray8', s='{}x{}'.format(width, height))
                .filter('fps', fps=framerate, round='up')
                .output(fn, pix_fmt='gray8', r=framerate, vcodec=vcodec)
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start ffmpeg process: {e}")

    for frame in images:
        try:
            process.stdin.write(frame.astype(np.uint8).tobytes())
        except Exception as e:
            raise RuntimeError(f"Failed to write frame to ffmpeg process: {e}")

    process.stdin.close()

    try:
        process.wait()
    except Exception as e:
        raise RuntimeError(f"ffmpeg process failed: {e}")

    return fn


def read_to_memmap(arr, ii, fl):
    arr[ii] = sitk.GetArrayFromImage(sitk.ReadImage(fl))
    arr.flush(); del arr
    if ii%10000==0: print(ii, flush=True)
    return


def listdir(pth, ifstring=None):
    """prints out complete path of list in directory

    Args:
        pth (_type_): _description_
        ifstring (_type_, optional): _description_. Defaults to None.

    Returns:
        list: list of items in directory with their complete path
    """
    if not ifstring==None:
        lst = [os.path.join(pth, xx) for xx in os.listdir(pth) if ifstring in xx]
    else:
        lst = [os.path.join(pth, xx) for xx in os.listdir(pth)]
    return lst