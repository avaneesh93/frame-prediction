import os
import numpy as np

save_path = os.path.dirname(os.getcwd()) + '/tune'

for ar in os.listdir(save_path):
    a = np.load(os.path.join(save_path,ar))
    print('{} {}'.format(ar, a[-1]))