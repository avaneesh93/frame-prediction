import os
from shutil import copy2, rmtree

def sort(frames_dir, seq):
    count = 0
    with open(seq) as sf:
        line = sf.readline()

        while line:
            # print(line)
            a,b,c,d,e = line.split()
            vid_name = a + '_uncomp.avi'
            folder_path = '{}/{}'.format(frames_dir, vid_name)
            # print(folder_path)
            if not os.path.exists(folder_path):
                line = sf.readline()
                continue

            count += 1
            dest_dir1 = '{}/walking/seq{}'.format(frames_dir, count)
            count += 1
            dest_dir2 = '{}/walking/seq{}'.format(frames_dir, count)
            count += 1
            dest_dir3 = '{}/walking/seq{}'.format(frames_dir, count)
            count += 1
            dest_dir4 = '{}/walking/seq{}'.format(frames_dir, count)

            b1,b2 = [int(i) for i in b.split('-')]
            c1,c2 = [int(i) for i in c.split('-')]
            d1,d2 = [int(i) for i in d.split('-')]
            e1,e2 = [int(i) for i in e.split('-')]

            os.makedirs(dest_dir1, exist_ok=True)
            os.makedirs(dest_dir2, exist_ok=True)
            os.makedirs(dest_dir3, exist_ok=True)
            os.makedirs(dest_dir4, exist_ok=True)

            for frame in os.listdir('{}/{}'.format(frames_dir, vid_name)):
                frame_count = frame[:-4]
                frame_count2 = int(frame_count)

                if frame_count2 >= b1 and frame_count2 <= b2:
                    copy2('{}/{}'.format(folder_path, frame), dest_dir1)
                elif frame_count2 >= c1 and frame_count2 <= c2:
                    copy2('{}/{}'.format(folder_path, frame), dest_dir2)
                elif frame_count2 >= d1 and frame_count2 <= d2:
                    copy2('{}/{}'.format(folder_path, frame), dest_dir3)
                elif frame_count2 >= e1 and frame_count2 <= e2:
                    copy2('{}/{}'.format(folder_path, frame), dest_dir4)

            rmtree('{}/{}'.format(frames_dir, vid_name), ignore_errors=True)

if __name__ == '__main__':
    frames_dir = '../datasets'
    seq = "../seq.txt"
    sort(frames_dir, seq)
