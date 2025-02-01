import sys
import os
import numpy as np
from datasets import load_dataset

mnist = load_dataset('mnist')
outputdir = sys.argv[1]
print("data out:", outputdir)


def save_data(name, img, label):
    imgfile = os.path.join(outputdir, f'{name}_img.idx')
    lblfile = os.path.join(outputdir, f'{name}_label.idx')

    imagearray = np.array([
        np.array(img, dtype=np.uint8).flatten() for image in img
        ], dtype=np.uint8)
    labelarray = np.array(label, dtype=np.uint8)

    imagearray.tofile(imgfile)
    labelarray.tofile(lblfile)


save_data("train",
          mnist["train"]["image"],
          mnist["train"]["label"]
          )
save_data("test",
          mnist["test"]["image"],
          mnist["test"]["label"]
          )
