import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)
out = np.loadtxt('output.dat', unpack=True)
plt.semilogy(out[0], out[1], label='data')
plt.semilogy(out[0], out[2], label='phys')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)
plt.show()


