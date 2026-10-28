import os
import shutil
import time
import subprocess
import csv
import numpy as np
import matplotlib.pyplot as plt
from ref_file2.femm import FemmSession

def dq_to_abc(dq0, b):
    clac2 = np.array([[1, -1/2, -1/2],
                    [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                    [1/2, 1/2, 1/2]])
    park_3x3 = np.array([
                [np.cos(b), np.sin(b), 0],
                [-np.sin(b), np.cos(b), 0],
                [0, 0, 1]
            ])
    abc = clac2.T @ park_3x3.T @ dq0
    return abc

print(dq_to_abc(np.array([[100], [0], [0]]), -np.pi/2))  # Example usage