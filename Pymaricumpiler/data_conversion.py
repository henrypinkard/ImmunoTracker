from pymaricumpiler import convert
import os
import csv
import numpy as np


data_list = '/LymphosightDatasets.csv'

#cosmos
imaris_dir = '/media/hugespace/henry/lymphosight/imaris_files'
raw_data_dir = '/media/hugespace/henry/lymphosight/raw_data/'
param_cache_dir = '/media/hugespace/henry/lymphosight/optimized_params/'

#Savio
imaris_dir = '/global/home/users/hbp/data/lymphosight/imaris_files'
raw_data_dir = '/global/home/users/hbp/data/lymphosight/raw_data/'
param_cache_dir = '/global/home/users/hbp/data/lymphosight/optimized_params/'


with open(data_list, 'r') as f:
    reader = csv.DictReader(f)
    experiments = list(reader)
    column_names = reader.fieldnames



def get_dataset_path(ID):
    ex =[e for e in experiments if e['ID'] == ID][0]
    path = '{} {}{}{}'.format(ex['Date'], ex['Folder'], os.sep, ex['Experiment'])
    return path

def get_value(ID, key):
    ex = [e for e in experiments if e['ID'] == ID][0]
    return ex[key]

ids = [e['ID'] for e in experiments]
#convert all with no corrections
for ID in ids:
    #TODO: apply to all
    if ID != '36':
        continue

    data_path = get_dataset_path(ID)
    magellan_dir = raw_data_dir + data_path

    print('\n\nconverting: {}\n'.format(magellan_dir))
    isr_ch = [int(v) for v in get_value(ID, 'ISR ch').split('+')]
    tp_ch = int(get_value(ID, 'TPR ch'))
    ntp = int(get_value(ID, 'Usable frames'))


    #TODO: remove
    ntp = 5

    convert(magellan_dir, position_registrations='optimize', input_filter_sigma=2,
            output_dir=imaris_dir, output_basename=ID, intra_stack_registration_channels=[1, 2, 3, 4, 5],
            timepoint_registration_channel=tp_ch, reverse_rank_filter=True, param_cache_dir=param_cache_dir,
            num_time_points=ntp, suffix='singularity_test',
            inter_stack_registration_channels=isr_ch, downsample_factor=2,
            stitch_regularization=0, stack=False, stitch=False, export=True)