import sys
import os
import struct
import numpy as np
from datasets import load_dataset

outputdir = sys.argv[1]
print("Data output directory:", outputdir)
if os.path.exists(outputdir):
    print("Data exists, skipping downloading")
    exit(0)

print("Downloading MNIST dataset")
mnist = load_dataset('mnist')


def save_data(name, img, label):
    print(f'Saving {name} dataset')
    imgfile = os.path.join(outputdir, f'{name}_img.idx')
    lblfile = os.path.join(outputdir, f'{name}_label.idx')
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    imagearray = np.array([
        np.array(image, dtype=np.uint8).flatten() for image in img
        ], dtype=np.uint8)
    labelarray = np.array(label, dtype=np.uint8)

    with open(imgfile, 'wb') as f:
        f.write(struct.pack(">I", 0x00000803))
        f.write(struct.pack(">I", len(imagearray)))
        f.write(struct.pack(">I", 28))
        f.write(struct.pack(">I", 28))
        imagearray.tofile(f)
        print(f'Saved {name} image set')

    with open(lblfile, 'wb') as f:
        f.write(struct.pack(">I", 0x00000801))
        f.write(struct.pack(">I", len(labelarray)))
        labelarray.tofile(f)
        print(f'Saved {name} label set')


save_data("train",
          mnist["train"]["image"],
          mnist["train"]["label"]
          )
save_data("test",
          mnist["test"]["image"],
          mnist["test"]["label"]
          )
