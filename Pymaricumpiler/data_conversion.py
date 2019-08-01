from pymaricumpiler import convert
import os
import csv
import numpy as np
from pathlib import Path
import sys
import argparse

#parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--stack', action='store_true')
parser.add_argument('--stitch', action='store_true')
parser.add_argument('--export', action='store_true')
parser.add_argument('--stack_lr', type=float, default=0.4)
parser.add_argument('--stitch_reg', type=float, default=0.0)
parser.add_argument('--ids', type=str, nargs='*')
parser.add_argument('--max_tp', type=int, default=-1)
parser.add_argument('--min_tp', type=int, default=-1)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--param_cache', type=str, default='optimized_params')
args = parser.parse_args()

# args = parser.parse_args(['--ids', '36', '--min_tp', '3', '--max_tp', '4',
#                           '--stitch_reg', '0.00001', '--stitch', '--ids', '36', '--max_tp', '5'])


print('Got arguments:')
print(args)

home = str(Path.home())

storage_path = home + '/lymphosight_data/'

imaris_dir = storage_path + 'imaris_files'
raw_data_dir = storage_path + 'raw_data/'
param_cache_dir = storage_path + args.param_cache + '/'
log_dir = storage_path + 'conversion_log/'
#make sure they all exist
for p in [imaris_dir, raw_data_dir, param_cache_dir, log_dir]:
    if not os.path.exists(p):
        os.makedirs(p)

data_list = '/opt/LymphosightDatasets.csv'

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

if len(args.ids) == 0:
    print('no ids arg, usign all')
    ids = [e['ID'] for e in experiments]
else:
    ids = args.ids
#convert all with no corrections
for ID in ids:

    data_path = get_dataset_path(ID)
    magellan_dir = raw_data_dir + data_path

    print('\nconverting ID: {} \t {}\n'.format(ID, magellan_dir))
    isr_ch = [int(v) for v in get_value(ID, 'ISR ch').split('+')]
    tp_ch = int(get_value(ID, 'TPR ch'))

    if (args.max_tp != -1):
        print('capping max_tp at: {}'.format(args.max_tp))
        max_tp = args.max_tp
    else:
        max_tp = int(get_value(ID, 'Usable frames'))

    if (args.min_tp != -1):
        print('capping min_tp at: {}'.format(args.min_tp))
        min_tp = args.min_tp
    else:
        min_tp = 0

    convert(magellan_dir,
            position_registrations='optimize',
            input_filter_sigma=2,
            output_dir=imaris_dir,
            output_basename=ID,
            intra_stack_registration_channels=[1, 2, 3, 4, 5],
            timepoint_registration_channel=tp_ch,
            reverse_rank_filter=True,
            param_cache_dir=param_cache_dir,
            log_dir=log_dir,
            min_tp=min_tp,
            max_tp=max_tp,
            suffix=args.suffix,
            stack_learning_rate=args.stack_lr,
            inter_stack_registration_channels=isr_ch,
            stitch_downsample_factor=2,
            stitch_regularization=args.stitch_reg,
            stack=args.stack,
            stitch=args.stitch,
            export=args.export)
