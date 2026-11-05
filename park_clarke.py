import os
import shutil
import time
import subprocess
import csv
import numpy as np
import matplotlib.pyplot as plt
from femm import FemmSession
import ctypes
from ctypes import wintypes



def dq_to_abc(dq0, b):
    sr32= np.sqrt(3)/2
    clac2 = np.array([[1, -1/2, -1/2],
                    [0, sr32, -sr32],
                    [1/2, 1/2, 1/2]])
    cb = np.cos(b)
    sb = np.sin(b)   
    park_3x3 = np.array([
                [cb, sb, 0],
                [-sb, cb, 0],
                [0, 0, 1]
            ])
    abc = clac2.T @ park_3x3.T @ dq0
    return abc
def dq_to_abc2(dq0, b):
    T = 2/3 * np.array([np.cos(b), np.cos(b - 2*np.pi/3), np.cos(b + 2*np.pi/3),
                            -np.sin(b), -np.sin(b - 2*np.pi/3), -np.sin(b + 2*np.pi/3),
                            1/2, 1/2, 1/2]).reshape(3, 3)
    abc = np.linalg.inv(T) @ dq0
    return abc
def abc_to_dq(i_abc, b):
    # 전체를 감싸는 바깥쪽 대괄호([ ])가 반드시 있어야 합니다.
    sr32= np.sqrt(3)/2
    clac_3x3 = 2/3 * np.array([
        [1, -1/2, -1/2], 
        [0, sr32, -sr32],
        [1/2, 1/2, 1/2]
    ])
    cb = np.cos(b)
    sb = np.sin(b)
    park_3x3 = np.array([
        [cb, sb, 0], 
        [-sb, cb, 0],
        [0, 0, 1]
    ])

    dq0 = park_3x3 @ clac_3x3 @ i_abc
    return dq0
def abc_to_dq2(i_abc, b):
    # 전체를 감싸는 바깥쪽 대괄호([ ])가 반드시 있어야 합니다.
    T = 2/3 * np.array([np.cos(b), np.cos(b - 2*np.pi/3), np.cos(b + 2*np.pi/3),
                        -np.sin(b), -np.sin(b - 2*np.pi/3), -np.sin(b + 2*np.pi/3),
                        1/2, 1/2, 1/2]).reshape(3, 3)
    dq0 = T @ i_abc

    return dq0

abc = [1, -3, 1]  # 테스트용 호출
print("abc:", abc)  # 테스트용 호출
dq = abc_to_dq(abc, np.pi/6)
print("dq:", dq)  # 테스트용 호출
print("dq_to_abc2:", dq_to_abc2(dq, np.pi/6))  # 테스트용 호출
print("dq_to_abc:", dq_to_abc(dq, np.pi/6))  # 테스트용 호출