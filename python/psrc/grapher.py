import sys
import matplotlib.pyplot as plt
import numpy as np

filepath = sys.argv[1]
file = open(filepath, "r")
filedata = file.read()
data = filedata.split(",")
data = list(map(float, data[:-1]))
x = np.arange(0, len(data))
y = np.array(data)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Training loss")
plt.show()
