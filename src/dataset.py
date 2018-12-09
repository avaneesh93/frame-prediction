import tensorflow as tf
import numpy as np
import os
import cv2
import glob
import pickle
from optical_flow_computation import *

class Dataset:

    def __init__(self, seq_dir):
        self.seq_dir = seq_dir
        self.seqs = []
        self.load()

    def load(self, pkl = '../datasets/set.pkl'):

        if os.path.exists(pkl):
            with open(pkl, 'rb') as f:
                self.seqs = pickle.load(f)
        else:        
            for seq in glob.iglob(self.seq_dir + '/seq*'):
                frames = []
                prefix = len(seq + '/')
                for image in sorted(glob.glob(seq+'/*.jpg'), key = lambda filename: int(filename[prefix:-4])):
                    frame = cv2.imread(image)[:, :, 0][:, :, np.newaxis]
                    frames.append(frame)

                self.seqs.append(frames)
            with open(pkl, 'wb') as f:
                pickle.dump(self.seqs, f)
        print("Loaded data")

    def set_values(self, batch_size, num_future_frames, num_past_frames = 5):
        self.pt = num_past_frames
        self.ft = num_future_frames
        self.batch_size = batch_size


    def __iter__(self):
        batches = None
        count = 0
        seqs2 = self.seqs[:50]
        for frames in seqs2:
            batch = np.array([frames[i - self.pt : i+self.ft+1] for i in range(self.pt, len(frames) - self.ft)])
            # print(batch.shape)
            if batches is None:
                batches = batch
            else:
                batches = np.concatenate((batches, batch))

            # print("frames {} done".format(count))
            count += 1
            if count > 50:
                break

        np.random.shuffle(batches)

        N = batches.shape[0]
        B = self.batch_size

        print(batches.shape)

        print("Returning iterator")
        return iter(batches[i:i+B] for i in range(0, N, B))

if __name__ == '__main__':
    ds = Dataset("../datasets/walking")
    ds.set_values(16, 1, 5)

    bs = 16
    pt = 5

    for t, X in enumerate(ds):
        x_np = np.array([X[index][pt] for index in range(bs)])
        y_np = np.array([X[index][-1] for index in range(bs)])
        ofp_np = np.array([Optical_Flow.compute(X[index][:pt+1]) for index in range(bs)])
        print(ofp_np.shape)
        break
