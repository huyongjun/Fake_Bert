import re
import numpy as np
import configparser as cp
import os
import crawler
import matplotlib.pyplot as plt



if __name__ == '__main__':
    data = []
    with open("data\epidemic_data.csv") as reader:
        for line in reader.readlines():
            data.append(line[:-1].split(",")[1])
    print(data)

    plt.plot(data)
    while True:
        plt.waitforbuttonpress()
        

