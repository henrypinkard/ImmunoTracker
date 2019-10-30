"""
Rewrite time calibration to compensate for bug in imaricumpiler
"""
import h5py
import numpy as np


source_name = '49.ims'
name = '48-49_fusion_zoomin.ims'
#add timestamps to fusion datasets
filename = '/Users/henrypinkard/Desktop/imaris_analysis/{}'
file = h5py.File(filename.format(name), mode='r+')
source_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/files_with_time_md/49_uncorrected.ims', mode='r+')

for key in file['DataSetInfo']['TimeInfo'].attrs.keys():
    if 'TimePoint' in key and 'Dataset' not in key and 'File' not in key:
        if key == 'TimePoint1':
            string = ''.join([letter.decode("utf-8") for letter in source_file['DataSetInfo']['TimeInfo'].attrs[key]])
            split1 = string.split(':')
            h = int(split1[0][-1])
            m = int(split1[1]) % 60
            s = float(split1[2]) %60
            reformatted = '2018-06-01 {:02d}:{:02d}:{:.3f}'.format(h, m, s)
            file['DataSetInfo']['TimeInfo'].attrs.create(key, np.array([c.encode('utf-8') for c in reformatted]))
        else:
            sourcekey = 'TimePoint{}'.format(int(key[9:]) - 1)
            string = ''.join([letter.decode("utf-8") for letter in source_file['DataSetInfo']['TimeInfo'].attrs[sourcekey]])
            split1 = string.split(':')
            h = int(split1[0][-1])
            m = int(split1[1]) % 60
            s = float(split1[2]) %60 + 0.001
            reformatted = '2018-06-01 {:02d}:{:02d}:{:.3f}'.format(h, m, s)
            file['DataSetInfo']['TimeInfo'].attrs.create(key, np.array([c.encode('utf-8') for c in reformatted]))

