from pymaricumpiler import convert
import os
import csv
import numpy as np

#download latest version of the master data spreadsheet
os.system("rclone copy wallercloud:henry/lymphosight/Datasets.csv .")

with open('Datasets.csv', 'r') as f:
    reader = csv.reader(f)
    list_form = list(reader)
str_array = np.array(list_form[1:])
data_indices = str_array[:, 0].astype(np.int)
os.system('rm Datasets.csv')

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

#test one
conversion_indices = np.array([36])
num_time_points = 4


#convert all with no corrections
imaris_dir = '/media/hugespace/henry/lymphosight/imaris_files'
for index in conversion_indices:
    data_path = get_dataset_path(index)
    namestring = get_dataset_name_string(index) + 'ch5 only strong register'
    magellan_dir = '/media/hugespace/henry/lymphosight/raw_data/' + data_path
    
    if os.path.isfile(imaris_dir + os.sep + namestring + '.ims'):
        print('skipping {} because its already converted'.format(namestring))
    else:
        print('\n\nconverting: {}\n'.format(magellan_dir))
        convert(magellan_dir, position_registrations='optimize', input_filter_sigma=2,
                output_dir=imaris_dir, output_basename=namestring, intra_stack_registration_channels=[1, 2, 3, 4, 5],
                inter_stack_registration_channels=[5], timepoint_registration_channel=5,
                reverse_rank_filter=True, optimization_log_dir='/media/hugespace/henry/lymphosight/optimization_logs/',
                num_time_points=5)


