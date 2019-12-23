#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from matplotlib import pyplot as plt

if __name__=='__main__':
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else :
        print('Usage: run_exe trajectories_file')
        print('format: x1 y1 theta1 x2 y2 theta2')
        sys.exit(0)

    data = open(file, "r")
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for line in data:
        # x1 y1 theta1 x2 y2 theta2
        value = [float(s) for s in line.split()]
        if len(value) == 6:
            x1.append(value[0])
            y1.append(value[1])
            x2.append(value[3])
            y2.append(value[4])
        else:
            continue

    p1, = plt.plot(x1, y1, 'b.')
    p2, = plt.plot(x2, y2, 'r.')
    plt.legend(handles=[p1, p2], labels=['Measurement', 'Ground Truth'])
    plt.show()
