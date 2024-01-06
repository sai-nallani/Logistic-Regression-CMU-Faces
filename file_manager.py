import os

# # parent, subdirs, files
# for parent, subdirs, files in os.walk('./faces'):
#     for file in files:
#         if not ('_2' in file or '_4' in file):
#             if ('happy' in file):
#                 os.rename(os.path.join(parent, file), f'./happy_dataset/happy/{file}')
#             elif ('sad' in file):
#                 os.rename(os.path.join(parent, file), f'./happy_dataset/sad/{file}')

import h5py
import numpy as np
from imageio.v2 import imread

file_names = []
happy_ds = []
for parent,subdirs,files in os.walk('./happy_dataset/happy'):
    for file in files:
        img_data = imread(os.path.join(parent, file))
        happy_ds.append(img_data)
        file_names.append(os.path.join(parent, file))

# happy_ds = np.array(happy_ds)

# with h5py.File("happyds.h5", "w") as file:
#     dset = file.create_dataset("x_train", data=happy_ds)
#     dset = file.create_dataset("y_train", data=np.ones(len(happy_ds)))

# ---------------------
    
sad_ds = []
for parent,subdirs,files in os.walk('./happy_dataset/sad'):
    for file in files:
        img_data = imread(os.path.join(parent, file))
        sad_ds.append(img_data)
        file_names.append(os.path.join(parent, file))

y = np.ones(len(happy_ds)).tolist()
y.extend(np.zeros(len(sad_ds)).tolist())
happy_ds.extend(sad_ds)

with h5py.File("happyds.h5", "w") as file:
    dset = file.create_dataset("x_train", data=np.array(happy_ds))
    dset = file.create_dataset("y_train", data= np.array(y))

print(file_names)
# image, row, col, rgb