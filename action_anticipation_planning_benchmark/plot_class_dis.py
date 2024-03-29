import matplotlib.pyplot as plt
import os

files = [
    # "./data/list_hmdb51_train_hmdb_ucf-feature.txt",
    "./data/list_ucf101_train_hmdb_ucf-feature.txt"
]

classes = []
for file in files:
    with open(file, "r") as f:
        lines = f.read().strip().split("\n")
    for l in lines:
        classes.append(int(l.split(" ")[-1]))

print(classes)
# draw classes distribution
cs = [0] * len(set(classes))
for c in classes:
    cs[c] += 1
print(cs)
# save plot
os.makedirs("./plot", exist_ok=True)

plt.savefig("./plot/class_dis.png")
