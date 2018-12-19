import os
import numpy as np

save_path = os.path.dirname(os.getcwd()) + '/tune'

"""
Prints the losses in the tune folder
"""
for ar in os.listdir(save_path):
    if ar[-3:] != 'npy':
        continue
    a = np.load(os.path.join(save_path,ar))
    print('{} {}'.format(ar, a[-1]))
