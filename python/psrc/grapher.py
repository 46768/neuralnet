import sys
import matplotlib.pyplot as plt
import numpy as np

filepath = sys.argv[1]
file = open(filepath, "r")
filedata = file.read()
data = filedata.split(",")

line_cnt = int(data[0])
print("line count:", line_cnt)
line_label = []
for i in range(line_cnt):
    line_label.append(data[1+i])
data = list(map(float, data[1+line_cnt:-1]))
print(len(data))
x = np.arange(0, len(data)//line_cnt)

fig, ax = plt.subplots()
for i in range(line_cnt):
    print(line_label[i])
    y = data[0+(i*(line_cnt-1))::line_cnt]
    ax.plot(x, y, label=line_label[i])
ax.legend()
ax.set_title("Network loss")
plt.show()
