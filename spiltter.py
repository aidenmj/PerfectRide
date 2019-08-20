import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'images'

negCls = '/img'


os.makedirs(root_dir +'/train' + negCls)



os.makedirs(root_dir +'/test' + negCls)

# Creating partitions of the data after shuffeling

currentCls = negCls

src = "images"+currentCls # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.8)])


train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]

test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))

print('Testing: ', len(test_FileNames))

# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, "images/train"+currentCls)
	


for name in test_FileNames:
    shutil.copy(name, "images/test"+currentCls)
