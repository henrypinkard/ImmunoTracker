from pymaricumpiler import convert
import os
import csv
import numpy as np
from pygellan import MagellanDataset

#download latest version of the master data spreadsheet
os.system("rclone copy wallercloud:henry/lymphosight/Datasets.csv .")

with open('Datasets.csv', 'r') as f:
  reader = csv.reader(f)
  list_form = list(reader)
str_array = np.array(list_form[1:])
data_indices = str_array[:, 0].astype(np.int)

def get_dataset_path(index):
    dat_list_index = int((data_indices == index).nonzero()[0])
    date = str_array[dat_list_index, 1]
    experiment = str_array[dat_list_index, 2]
    dataset = str_array[dat_list_index, 3]
    path = '{} {}{}{}'.format(date, experiment, os.sep, dataset)
    return path

def get_dataset_name_string(index):
    dat_list_index = int((data_indices == index).nonzero()[0])
    date = str_array[dat_list_index, 1]
    experiment = str_array[dat_list_index, 2]
    dataset = str_array[dat_list_index, 3]
    path = '{}_{}{}{}'.format(date, experiment, '_', dataset)
    return path


#convert all with no corrections
imaris_dir = '/media/hugespace/henry/data/lymphosight/imaris_files_uncorrected'
for index in data_indices[data_indices >= 9]:
    data_path = get_dataset_path(index)
    namestring = get_dataset_name_string(index) + '_uncorrected'
    magellan_dir = '/media/hugespace/henry/data/lymphosight/raw_data/' + data_path
    print('\n\nconverting: {}\n'.format(magellan_dir))
    convert(magellan_dir, do_intra_stack=False, do_inter_stack=False, do_timepoints=False,
                output_dir=imaris_dir, output_basename=namestring, intra_stack_registration_channels=[1, 2, 3, 4, 5],
                intra_stack_noise_model_sigma=2, intra_stack_zero_center_sigma=3,
                intra_stack_likelihood_threshold_smooth=1.0, intra_stack_likelihood_threshold=-18,
                inter_stack_registration_channel=0, inter_stack_max_z=7, timepoint_registration_channel=0, n_cores=24)

