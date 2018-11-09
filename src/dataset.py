import tensorflow as tf
import numpy as np
import os
import cv2
import glob
import pickle

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
                for image in sorted(glob.glob(seq+'/*.jpg')):
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
        for frames in self.seqs:
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

        N = batches.shape[0]
        B = self.batch_size

        print(batches.shape)

        return iter(batches[i:i+B] for i in range(0, N, B))

if __name__ == '__main__':
    ds = Dataset("../datasets/walking")
    ds.set_values(16, 1, 5)

    for t, x in enumerate(ds):
        print(x.shape)
        break
