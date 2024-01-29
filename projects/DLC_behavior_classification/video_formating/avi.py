import SimpleITK as sitk
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