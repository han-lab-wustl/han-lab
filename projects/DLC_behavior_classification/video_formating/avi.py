import SimpleITK as sitk, os
import numpy as np
import ffmpeg


def vidwrite(fn, images, framerate=31.25*2, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', r=framerate, pix_fmt='gray8', s='{}x{}'.format(width, height))
            .filter('fps', fps=framerate, round='up')
            .output(fn, pix_fmt='gray8', r=framerate, vcodec=vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()
    
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