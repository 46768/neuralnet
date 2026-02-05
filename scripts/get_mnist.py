#!/usr/bin/python3
import argparse
import os
import struct
import numpy as np
from datasets import load_dataset

outputdir = ""
cli = argparse.ArgumentParser(
        prog="MnistGetter",
        description="Fetch and write MNIST data in a format this library understands",)
cli.add_argument("-o", "--output", required=True)

args = cli.parse_args()

if not os.path.exists(args.output):
    os.mkdir(args.output)

print("Downloading MNIST dataset")
mnist = load_dataset('mnist')


def save_data(name, img, label):
    print(f'Saving {name} dataset')
    binfile = os.path.join(args.output, f'{name}_dataset.bin')

    print("Flattening image")
    imagearray = np.array([
        np.array(image, dtype=np.uint8).flatten() for image in img
        ], dtype=np.uint8)

    print("Fetching label")
    labelarray = np.array(label, dtype=np.uint8)

    if len(imagearray) != len(labelarray):
        print("Error: Mismatch between label size and image size, aborting write")
        return

    print("Writing")
    with open(binfile, 'wb') as f:
        input_size = len(imagearray[0])

        # Write dataset headers
        f.write(struct.pack("@I", len(imagearray)))  # Dataset size
        f.write(struct.pack("@I", input_size))  # Input size
        f.write(struct.pack("@I", 10))  # Output size

        # Write each data pairs
        for i in range(len(imagearray)):
            # Write input data
            normalized_img = [float(b) / 255 for b in imagearray[i]]
            for img_byte in normalized_img:
                f.write(struct.pack("@f", img_byte))

            # Write output data
            normalized_label = [float(int(labelarray[i] == x)) for x in range(10)]
            for lbl_byte in normalized_label:
                f.write(struct.pack("@f", lbl_byte))


save_data("train",
          mnist["train"]["image"],
          mnist["train"]["label"]
          )
save_data("test",
          mnist["test"]["image"],
          mnist["test"]["label"]
          )
