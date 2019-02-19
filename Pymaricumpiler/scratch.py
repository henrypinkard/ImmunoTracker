from imariswriter import ImarisJavaWrapper
import os
import numpy as np

directory = os.getcwd()
name = 'imaristest'
size = [40, 400, 400]
num_frames = 10
with ImarisJavaWrapper(directory, name, size, 1, 1, num_frames, 0.5,0.5) as writer:
    for time_index in range(num_frames):
        data = np.random.randint(0, 255, size, dtype=np.uint8)
        for z_index, image in enumerate(data):
            print(time_index, z_index)
            writer.write_z_slice(image, z_index, 0, time_index, 0)
print('finished')