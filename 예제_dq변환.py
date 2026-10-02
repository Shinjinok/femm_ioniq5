import os
import shutil
import time
import datetime
import numpy as np
import pandas as pd
import femm
from multiprocessing import Pool, cpu_count

def abc_to_dq(i_abc, b):
  
  #dq = np.array([0, 0])
  clac=2/3*np.array([[1, -1/2, -1/2], 
                     [0,np.sqrt(3)/2, -np.sqrt(3)/2]])
  park=np.array([[np.cos(b), np.sin(b)], 
                 [-np.sin(b), np.cos(b)]])
  dq= park@clac@i_abc
  return dq


def dq_to_abc(v_dq, b):
  clac=np.array([[1, -1/2, -1/2], 
                 [0,np.sqrt(3)/2, -np.sqrt(3)/2]])
  iclac=np.transpose(clac)
  ipark=np.array([[np.cos(b), -np.sin(b)], 
                  [np.sin(b), np.cos(b)]])
  v=iclac@ipark@v_dq
  return v

abc = np.array([0.0289959, -0.0154502,-0.0154569])*8

dq= abc_to_dq(abc,0)
print(dq)

