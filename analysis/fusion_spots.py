"""
Add in spots to fusion dataset from constituents
"""
import h5py
import numpy as np
from scipy import io


def read_spots(file):
    spots = {}
    for folder_name in file['Scene']['Content']:
        if 'Points' not in folder_name:
            continue
        object = file['Scene']['Content'][folder_name]
        object_name = object.attrs['Name'][0].decode('utf-8')
        coords = object['CoordsXYZR']
        time_indices = np.ravel(object['Time'])
        if 'Edges' in object:
            edges = object['Edges']
        else:
            edges = None
        spots[object_name + '_coords'] = np.array(coords)
        spots[object_name + '_time_indices'] = np.array(time_indices)
        if edges is not None:
            spots[object_name + '_edges'] = np.array(edges)
    return spots

filepath48 = '/Users/henrypinkard/Desktop/imaris_analysis/48.ims'
filepath49 = '/Users/henrypinkard/Desktop/imaris_analysis/49.ims'
filepath4849 = '/Users/henrypinkard/Desktop/imaris_analysis/48-49_fusion_zoomin.ims'

# 44-45: [  -2, 1102,  587]
# 48-49: [-10 692 857]


file48 = h5py.File(filepath48, mode='r')
file49 = h5py.File(filepath49, mode='r')
file4849 = h5py.File(filepath4849, mode='r+')

spots48 = read_spots(file48)
spots49 = read_spots(file49)

all_spots = {}
for name in ['GFP', 'RFP', 'VPD', 'XCR1']:
    coords48 = spots48['{}_coords'.format(name)] + np.array([[0, 0, 40, 0]])
    coords49 = spots49['{}_coords'.format(name)] + np.array([[857, 692, 0, 0]]) * 0.555
    all_spots['{}_coords'.format(name)] = np.concatenate([coords48, coords49], axis=0)

    time48 = spots48['{}_time_indices'.format(name)]
    time49 = spots49['{}_time_indices'.format(name)] + 1
    all_spots['{}_time_indices'.format(name)] = np.concatenate([time48, time49], axis=0)

    if '{}_edges'.format(name) in spots49:
        edges = spots49['{}_edges'.format(name)] + coords48.shape[0]
        all_spots['{}_edges'.format(name)] = edges


io.savemat('/Users/henrypinkard/Desktop/imaris_analysis/48_49_fusion_spots.mat', all_spots)

#add into new one
# file4849.create_group('Scene')
# file4849['Scene'].create_group('Content')
# file4849['Scene']['Content'].attrs.create('NumberOfPoints', 4)


# file4849['Scene']['Content'].create_group('Points0')
# coords, time_indices, edges, timeInfo, attrs = spots48['XCR1']
# for at_name in list(attrs):
#     file4849['Scene']['Content']['Points0'].attrs.create(at_name, attrs[at_name])
# file4849['Scene']['Content']['Points0'].create_dataset('Coords', data=coords)
# file4849['Scene']['Content']['Points0'].create_dataset('Time', data=time_indices)
# file4849['Scene']['Content']['Points0'].create_dataset('TimeInfos', data=timeInfo)
# if edges is not None:
#     file4849['Scene']['Content']['Points0'].create_dataset('Edges', data=edges)