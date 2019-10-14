"""
Rewrite time calibration to compensate for bug in imaricumpiler
"""
import h5py
import numpy as np

names = [ '21.ims', '24.ims', '25.ims', '27.ims', '28.ims', '31.ims', '34.ims', '36.ims', '38.ims', '39.ims', '44.ims',
  '45.ims', '46.ims', '47.ims', '48-49_fusion_zoomin.ims', '48.ims', '49.ims', '49smoothed_but_no_corrections.ims',
  '51.ims', '52.ims', '53.ims', '54.ims'];

for name in names:
    print(name)
    filename = '/Users/henrypinkard/Desktop/imaris_analysis/{}'.format(name)
    file = h5py.File(filename, mode='r+')

    for key in file['DataSetInfo']['TimeInfo'].attrs.keys():
        if 'TimePoint' in key and 'Dataset' not in key and 'File' not in key:
            string = ''.join([letter.decode("utf-8") for letter in file['DataSetInfo']['TimeInfo'].attrs[key]])

            split1 = string.split(':')
            h = int(split1[0][-1])
            m = int(split1[1]) % 60
            s = float(split1[2]) %60
            reformatted = '2018-06-01 {:02d}:{:02d}:{:.3f}'.format(h, m, s)
            # print(key + '\t' + string)
            # print(reformatted)
            file['DataSetInfo']['TimeInfo'].attrs.create(key, np.array([c.encode('utf-8') for c in reformatted]))

