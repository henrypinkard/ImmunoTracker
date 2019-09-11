import csv
from pathlib import Path
import os
from data_reading import open_magellan, read_raw_data
from PIL import Image
import numpy as np
import napari
from stitcher import stitch_single_channel, apply_intra_stack_registration


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

img = p_zyxc_stacks[0][60]
img = img[::2, ::2]
sequence = []
for i in range(img.shape[0] * 4):
    print(i)
    newimg = np.zeros_like(img)
    newimg[:i // 4] = img[:i//4]
    if (i // 4) % 2 == 0:
        newimg[i // 4, :i%4] = img[i // 4, :i%4]
    else:
        newimg[i // 4, -i % 4:] = img[i // 4, -i % 4:]
    sequence.extend([c.T for c in newimg.T])

exporttiffstack(np.stack(sequence),  '/Users/henrypinkard/Desktop/', 'raster_demo')
pass

# home = str(Path.home())
#
# storage_path = home + '/lymphosight_data/'
# param_cache_dir = storage_path + 'optimized_params' + '/'
# log_dir = storage_path + 'conversion_log/'
# param_cache_name = str(ID) + '_tp{}'.format(0)
# saved_name = '{}{}_optimized_params.npz'.format(param_cache_dir, param_cache_name)
# if os.path.isfile(saved_name):
#     with np.load(saved_name) as loaded:
#         if 'p_yx_translations' in loaded:
#             print('Loaded params from: ' + saved_name)
#             p_yx_translations = loaded['p_yx_translations']
#
# for pos_index in p_zyxc_stacks.keys():
#     exporttiffstack(np.reshape(np.moveaxis(p_zyxc_stacks[pos_index], 3, 0), [-1, p_zyxc_stacks[0].shape[1], p_zyxc_stacks[0].shape[2]]),
#                     '/Users/henrypinkard/Desktop/', 'intra_stack_uncorrected_{}'.format(pos_index))
#     p_zyxc_stacks[pos_index] = np.stack([apply_intra_stack_registration(p_zyxc_stacks[pos_index][..., c], p_yx_translations[pos_index],
#                                                               background=0, mode='float') for c in range(6)], axis=3)
#
#     exporttiffstack(np.reshape(np.moveaxis(p_zyxc_stacks[pos_index], 3, 0), [-1, p_zyxc_stacks[0].shape[1], p_zyxc_stacks[0].shape[2]]).astype(np.uint8),
#                     '/Users/henrypinkard/Desktop/', 'intra_stack_corrected_{}'.format(pos_index))



# image = p_zyxc_stacks[0][40, ..., 0]
# image1 = image[:-60, 60:].astype(np.float)
# image2 = image[60:, :-60].astype(np.float)
#
# fft2 = np.fft.fft2(image2.astype(np.float))
# fft1 = np.fft.fft2(image1.astype(np.float))
#
# phasecorr = np.fft.fftshift(np.abs(np.fft.ifft2((fft1 * fft2.conj()) / np.abs(fft1 * fft2.conj()))))
#
# phasecorr = phasecorr / np.max(phasecorr) * 65535
# exporttiffstack(phasecorr.astype(np.uint16), '/Users/henrypinkard/Desktop/', 'phasecorr')
#
#
# with napari.gui_qt():
#     # create the viewer with four layers
#     viewer = napari.view( xcorr)





# exporttiffstack(np.reshape(np.moveaxis(p_zyxc_stacks[1], 3, 0), [-1, p_zyxc_stacks[0].shape[1], p_zyxc_stacks[0].shape[2]]), '/Users/henrypinkard/Desktop/')

