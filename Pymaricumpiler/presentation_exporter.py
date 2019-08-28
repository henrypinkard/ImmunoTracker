import csv
from pathlib import Path
import os
from data_reading import open_magellan, read_raw_data
from PIL import Image
import numpy as np
import napari

def exporttiffstack(datacube, path, name='export'):
    '''
    Save 3D numpy array as a TIFF stack
    :param datacube:
    '''
    if len(datacube.shape) == 2:
        imlist = [Image.fromarray(datacube)]
    else:
        imlist = []
        for i in range(datacube.shape[0]):
            imlist.append(Image.fromarray(datacube[i,...]))
    path = "{}{}.tif".format(path, name)
    imlist[0].save(path, compression="tiff_deflate", save_all=True, append_images=imlist[1:])



ID = '36'

home = str(Path.home())
storage_path = home + '/lymphosight_data/'
raw_data_dir = storage_path + 'raw_data/'

data_list = home + '/GitRepos/LymphoSight/LymphosightDatasets.csv'

with open(data_list, 'r') as f:
    reader = csv.DictReader(f)
    experiments = list(reader)
    column_names = reader.fieldnames

def get_dataset_path(ID):
    ex = [e for e in experiments if e['ID'] == ID][0]
    path = '{} {}{}{}'.format(ex['Date'], ex['Folder'], os.sep, ex['Experiment'])
    return path

def get_value(ID, key):
    ex = [e for e in experiments if e['ID'] == ID][0]
    return ex[key]


data_path = get_dataset_path(ID)
magellan_dir = raw_data_dir + data_path

magellan, metadata = open_magellan(magellan_dir)

p_zyxc_stacks, nonempty_pixels, timestamp = read_raw_data(magellan, metadata, time_index=0,
                                                          reverse_rank_filter=True, input_filter_sigma=2)


image = p_zyxc_stacks[0][40, ..., 0]


image1 = image[:-60, 60:].astype(np.float)
image2 = image[60:, :-60].astype(np.float)

fft2 = np.fft.fft2(image2.astype(np.float))
fft1 = np.fft.fft2(image1.astype(np.float))

phasecorr = np.fft.fftshift(np.abs(np.fft.ifft2((fft1 * fft2.conj()) / np.abs(fft1 * fft2.conj()))))

phasecorr = phasecorr / np.max(phasecorr) * 65535

exporttiffstack(phasecorr.astype(np.uint16), '/Users/henrypinkard/Desktop/', 'phasecorr')


with napari.gui_qt():
    # create the viewer with four layers
    viewer = napari.view( xcorr)



exporttiffstack(image1, '/Users/henrypinkard/Desktop/', 'image1')
exporttiffstack(image2, '/Users/henrypinkard/Desktop/', 'image2')

exporttiffstack(np.abs(np.fft.fftshift(np.fft.fft2(image1.astype(np.float)))).astype(np.uint16), '/Users/henrypinkard/Desktop/', 'image1fft')
exporttiffstack(np.abs(np.fft.fftshift(np.fft.fft2(image2.astype(np.float)))).astype(np.uint16), '/Users/henrypinkard/Desktop/', 'image2fft')




exporttiffstack(np.reshape(np.moveaxis(p_zyxc_stacks[1], 3, 0), [-1, p_zyxc_stacks[0].shape[1], p_zyxc_stacks[0].shape[2]]), '/Users/henrypinkard/Desktop/')


pass

#TODO: easy way of getting individual slices and orthogonal projections

# import napari
# with napari.gui_qt():
#     # create the viewer with four layers
#     viewer = napari.view(p_zyxc_stacks[0][0])
#     # remove a layer
#     viewer.layers.append(p_zyxc_stacks[0][0])

