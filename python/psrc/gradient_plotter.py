import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import sys


filepath = sys.argv[1]
file = open(filepath, "r")
filedata = file.read()
data = filedata.split(",")

layer_cnt = int(data[0])
layer_size = list(map(int, data[1:layer_cnt+1]))
layer_size_padded = list(
        map(lambda x: (x+7) & ~7, map(int, data[1:layer_cnt+1]))
        )
weight_gradient_size = 0
gradient_data_size = 0
for i in range(layer_cnt - 1):
    gradient_data_size += layer_size_padded[i] * layer_size_padded[i+1]
    weight_gradient_size += layer_size_padded[i] * layer_size_padded[i+1]
    gradient_data_size += layer_size_padded[i+1]

gradient_upt_cnt = len(data[layer_cnt+1:]) // gradient_data_size
gradient_data = data[layer_cnt+1:]

weight_gradient_set = []
bias_gradient_set = []

for i in range(gradient_upt_cnt):
    data_offset = gradient_data_size * i
    weight_gradient = []
    bias_gradient = []

    for j in range(layer_cnt-1):
        pass
