#!/usr/bin/env python

import csv
import sys
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt


def parse_mean(t, t_min, t_max):
    t_other = t[t[:, 1] < t_min]
    t_contact = t[t[:, 1] >= t_min]
    t_contact = t_contact[t_contact[:, 1] <= t_max]
    t_contact_v = t_contact[1:, 0] - t_contact[:-1, 0]
    t_other_v = t_other[1:, 0] - t_other[:-1, 0]
    return (np.mean(t_contact_v), np.mean(t_other_v))


def parse_mean_filt(t, t_min, t_max, kernel=11):
    t_other = t[t[:, 1] < t_min]
    t_contact = t[t[:, 1] >= t_min]
    t_contact = t_contact[t_contact[:, 1] <= t_max]
    t_contact_v = sig.medfilt(t_contact[1:, 0] - t_contact[:-1, 0], kernel)
    t_other_v = sig.medfilt(t_other[1:, 0] - t_other[:-1, 0], kernel)
    return (np.mean(t_contact_v), np.mean(t_other_v))


def parse_t(f):
    reader = csv.reader(f, delimiter=" ")
    t = None
    reader.next()
    for row in reader:
        t_real = float(row[1])
        t_sim = float(row[0])
        row_array = np.array([[t_real, t_sim]])
        if t is not None:
            t = np.vstack((t, row_array))
        else:
            t = row_array
    return t

fn = sys.argv[1]
f = open(fn, 'r')
t = parse_t(f)

if(len(sys.argv) == 4):
    t_min = float(sys.argv[2])
    t_max = float(sys.argv[3])
    (mean_contact, mean_other) = parse_mean(t, t_min, t_max)
    (mean_contact_filt, mean_other_filt) = parse_mean_filt(t, t_min, t_max)
    print "Mean of Processing Time during Contact:", mean_contact
    print "Mean of Processing Time, not in Contact:", mean_other
    print "Mean of Filtered Processing Time during Contact:", mean_contact_filt
    print "Mean of Filtered Processing Time, not in Contact:", mean_other_filt

plt.plot(t[1:, 1], sig.medfilt(t[1:, 0] - t[:-1, 0], 11))
plt.show()
