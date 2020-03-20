from PIL import Image
import glob
from os.path import expanduser
import numpy as np

def read_cropped_image( filename ):
    f = Image.open( filename )
    crop =  f.crop( (15,40,15+148-1,40+148-1))
    newsize = crop.resize( (64,64 ) )
    return newsize

def read_images( folder=None, max_no=1000, add_noise=True ):
    if folder is None:
        folder = expanduser("~") + "/ImageDataSets/CelebA/img_align_celeba"
    files = glob.glob(folder + "/*.jpg")
    lsamples = [ np.asarray(read_cropped_image( filename ) ) for filename in files[:max_no] ]
    samples = np.array( lsamples ).astype( np.float32 )
    deq = samples/256. + np.random.uniform( low=-0.01,high=0.01, size=[max_no,64,64,3]).astype( np.float32)
    if add_noise:
        ret = deq
    else:
        ret = samples/255.
    return ret

