import numpy as np

b=np.radians(90)
i=np.array([100, 150, -270])
clac=2/3*np.array([[1, -1/2, -1/2], [0,np.sqrt(3)/2, -np.sqrt(3)/2]])
park=np.array([[np.cos(b), np.sin(b)], [-np.sin(b), np.cos(b)]])
dq= park@clac@i
print(f"dq:{dq}")

iclac=np.transpose(np.array([[1, -1/2, -1/2], [0,np.sqrt(3)/2, -np.sqrt(3)/2]]))
ipark=np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
iabc=iclac@ipark@dq  
print(f"iabc:{iabc}")
