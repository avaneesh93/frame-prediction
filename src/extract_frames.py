import os
import cv2

def extract(vids_dir, frames_dir):
    
    for vid in os.listdir(vids_dir):
        vidcap = cv2.VideoCapture('{}/{}'.format(vids_dir, vid))
        if not os.path.exists('{}/{}'.format(frames_dir, vid)):
            os.makedirs('{}/{}'.format(frames_dir, vid))


        success,image = vidcap.read()
        # print(success)
        count = 1
        while success:
          filepath = '{}/{}/{}.jpg'.format(frames_dir, vid, count)
          image = image[0:120, 20:140]
          cv2.imwrite(filepath, image)     
          success,image = vidcap.read()
          if not success:
            break
          #print('Read a new frame: ', success)
          count += 1

if __name__ == '__main__':
    vids_dir = '/Users/avaneesh/CS682/project/walking'
    frames_dir = '../datasets/'
    extract(vids_dir, frames_dir)