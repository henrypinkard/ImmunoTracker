from pymaricumpiler import convert
import os
import csv
import numpy as np


#download up to date spreadsheet listing experiments
if 0 == os.system("rclone copy wallercloud:henry/lymphosight/LymphosightDatasets.csv ."):
    print('downloading dataset list')
    data_list = './LymphosightDatasets.csv'
    print('data list downloaded')
    imaris_dir = '/media/hugespace/henry/lymphosight/imaris_files'
    raw_data_dir = '/media/hugespace/henry/lymphosight/raw_data/'
    param_cache_dir = '/media/hugespace/henry/lymphosight/optimized_params/'
else:
    #on the berkeley cluster 
    data_list = '/global/home/users/hbp/data/lymphosight/LymphosightDatasets.csv'
    imaris_dir = '/global/home/users/hbp/data/lymphosight/imaris_files'
    raw_data_dir = '/global/home/users/hbp/data/lymphosight/raw_data/'
    param_cache_dir = '/global/home/users/hbp/data/lymphosight/optimized_params/'

with open('LymphosightDatasets.csv', 'r') as f:
    reader = csv.reader(f)
    list_form = list(reader)
str_array = np.array(list_form[1:])
data_indices = str_array[:, 0].astype(np.int)
#remove data list if its beign redownloaded every time
if data_list == './LymphosightDatasets.csv':
    os.remove(data_list)

def get_dataset_path(ID):
    dat_list_index = int((data_indices == ID).nonzero()[0])
    date = str_array[dat_list_index, 1]
    experiment = str_array[dat_list_index, 2]
    dataset = str_array[dat_list_index, 3]
    path = '{} {}{}{}'.format(date, experiment, os.sep, dataset)
    return path

def get_dataset_name_string(ID):
    dat_list_index = int((data_indices == ID).nonzero()[0])
    date = str_array[dat_list_index, 1]
    experiment = str_array[dat_list_index, 2]
    dataset = str_array[dat_list_index, 3]
    path = '{}_{}{}{}'.format(date, experiment, '_', dataset)
    return path

#test one
conversion_ID = np.array([36])
num_time_points = 5


#convert all with no corrections
for ID in conversion_ID:
    data_path = get_dataset_path(ID)
    magellan_dir = raw_data_dir + data_path

    # if os.path.isfile(imaris_dir + os.sep + namestring + '.ims'):
    #     print('skipping {} because its already converted'.format(namestring))
    # else:
    print('\n\nconverting: {}\n'.format(magellan_dir))
    convert(magellan_dir, position_registrations='optimize', input_filter_sigma=2,
            output_dir=imaris_dir, output_basename=ID, intra_stack_registration_channels=[1, 2, 3, 4, 5],
            timepoint_registration_channel=5, reverse_rank_filter=True, param_cache_dir=param_cache_dir,
            num_time_points=num_time_points,
            suffix='2x2_ch05_noreg', inter_stack_registration_channels=[0, 5], downsample_factor=2,
            stitch_regularization=0, stack=False, stitch=False, export=True)