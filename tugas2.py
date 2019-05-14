import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def importDataSet():
    data = []
    with open("Tugas 2 ML Genap 2018-2019 Dataset Tanpa Label.csv") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            data.append([float(row[0]), float(row[1])])
    return np.array(data)


def Euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def Manhattan(x, y):
    return np.abs(x[0] - y[0]) + np.abs(x[1] - y[1])


def VektorTerdekat(vmap, p, col, row):
    bmu = np.array([0, 0])
    minimum_dist = Euclidean(vmap[0][0], p)
    for x in range(col):
        for y in range(row):
            dist = Euclidean(vmap[x][y], p)
            if dist < minimum_dist:
                minimum_dist = dist
                bmu = np.array([x, y])
    return bmu, minimum_dist


def Cluster(vmap, data, col, row):
    clas = []
    for p in data:
        bmu, _ = VektorTerdekat(vmap, p, col, row)
        clas.append([p, bmu[0] + bmu[1] * col])
    return clas


def Show(cluster, vmap, col, row):
    color = [
        "cyan",
        "yellow",
        "red",
        "gold",
        "darkgreen",
        "maroon",
        "darkgray",
        "orange",
        "orchid",
        "sienna",
        "chocolate",
        "lavender",
        "cerise",
    ]
    i = 1
    for v in np.reshape(vmap, (col * row, 2)):
        plt.text(
            v[0],
            v[1],
            i,
            bbox={"facecolor": "red", "alpha": 0.5, "pad": 1},
            fontweight="bold",
        )
        i += 1
    i = 0
    for c in cluster:
        plt.plot(c[0][0], c[0][1], color=color[c[1]], marker="+", linestyle="None")
    plt.show()


def isStop(data):
    for d in data:
        if d < np.avercage(data) - 1:
            return False
    return True


if __name__ == "__main__":
    data_set = importDataSet()
    data = data_set.T

    col = 4
    row = 3
    dim = 2
    vmap = np.random.uniform(0, 1, (col, row, dim))
    lr = 1
    sigma = 2

    m = np.reshape(vmap, (col * row, dim)).T

    # Training
    for t in range(20):
        for p in data_set:
            bmu, min_dist = VektorTerdekat(vmap, p, col, row)
            for x in range(col):
                for y in range(row):
                    if Manhattan(np.array([bmu[0], bmu[1]]), np.array([x, y])) < 4:
                        theta = np.exp(
                            -(Euclidean(bmu, np.array([x, y])) ** 2)
                            / (2 * (sigma ** 2))
                        )
                        delta_w = lr * theta * (p - vmap[x][y])
                        vmap[x][y] = vmap[x][y] + delta_w

        lr = 0.1 * np.exp(-t / 12)
        sigma = 2 * np.exp(-t / 12)
    clas = Cluster(vmap, data_set, col, row)

    # Show Cluster
    Show(clas, vmap, col, row)

