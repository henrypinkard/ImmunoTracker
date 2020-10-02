from util import *
import napari

denisty_file = h5py.File('/Users/henrypinkard/Desktop/imaris_analysis/density_maps2.hdf', 'r')

pass

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(denisty_file['45_xcr1_within_100_tp_0'], name='t_0')
    # viewer.add_image(denisty_file['45_xcr1_within_100_tp_19'], name='t_19')
    # viewer.add_image(np.array(denisty_file['45_xcr1_within_100_tp_19']
    #                           ) / (1 + np.array(denisty_file['45_xcr1_within_100_tp_0'])), name='t_19')
