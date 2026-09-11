import matplotlib.pyplot as plt
import numpy as np

# Dataset 1
x1 = [0, 0.19, 0.50, 0.91, 1.26, 1.45, 1.55, 1.63, 1.73, 1.83]
y1 = [0.00, 100, 300, 500, 800, 1000, 1500, 5000, 10000, 20000]

# Dataset 2
x2 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8]
y2 = [0, 18, 35, 60, 95, 150, 270, 600, 1200, 2000, 3800, 7000]

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, marker='o', label='Dataset 1')
plt.plot(x2, y2, marker='s', label='Dataset 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparison of Two Datasets')
plt.legend()
plt.grid(True)
plt.savefig('plot.png', bbox_inches='tight')
plt.close()
print("Plot generated successfully.")