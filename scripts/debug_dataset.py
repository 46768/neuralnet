#!/usr/bin/python3
import argparse
import os
import struct
import pathlib
from PIL import Image
import numpy as np

outputdir = ""
cli = argparse.ArgumentParser(
        prog="DatasetDebugger",
        description="Debug dataset binary files",)
cli.add_argument("-i", "--input", required=True, help="Input path to dataset file")
cli.add_argument("-o", "--output", help="Output path for specific flags")
cli.add_argument("-c", "--count", default=5, type=int, help="Number of pairs to print out, default to 5, set to -1 for all, if more than number of pairs, will printout all the pairs")
cli.add_argument("-d", "--dump", action="store_true", help="Whether to dump the content or output in pairs")
cli.add_argument("-p", "--picture", action="store_true", help="Write as image instead of floats")

args = cli.parse_args()

if not os.path.exists(args.input):
    print("Error: file not found")
    exit(1)

with open(args.input, 'rb') as f:
    file_buffer = f.read()

    dsize = struct.unpack("@I", file_buffer[:4])[0]
    isize = struct.unpack("@I", file_buffer[4:8])[0]
    osize = struct.unpack("@I", file_buffer[8:12])[0]

    print(dsize)
    print(isize)
    print(osize)

    if args.picture:
        if args.output is None:
            print("Missing Args: output")
            exit(1)

        path = pathlib.Path(args.output) / "picdump"
        if not os.path.exists(path):
            os.mkdir(path)
        for i in range(args.count):
            offset = 12 + (i * 4 * (isize + osize))
            idata = []

            for j in range(isize):
                idata.append(struct.unpack("@f", file_buffer[offset+(4*j):offset+4+(4*j)])[0])

            #npidata = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            npidata = np.array(idata, dtype=np.float32).reshape((28, 28))
            npidata = (npidata * 255 / np.max(npidata)).astype(np.uint8)
            im = Image.fromarray(npidata)
            fpath = path / (f'idx{i}.png')
            im.save(fpath, format="png")
    elif args.dump:
        for i in range((len(file_buffer)-12) // 4):
            print(struct.unpack("@f", file_buffer[12+(i*4):12+(i*4)+4])[0], end=" ")
    else:
        for i in range(args.count):
            offset = 12 + (i * 4 * (isize + osize))
            idata = []
            odata = []

            for j in range(isize):
                idata.append(struct.unpack("@f", file_buffer[offset+(4*j):offset+4+(4*j)])[0])

            for j in range(osize):
                odata.append(struct.unpack("@f", file_buffer[offset+(4*j)+(isize*4):offset+4+(4*j)+(4*isize)])[0])

            print(f'Index {i}:')
            print(idata)
            print(odata)
