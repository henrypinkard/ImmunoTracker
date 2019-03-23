import numpy as np

def x_corr_register_3D(volume1, volume2, max_shift):
    """
    Compute cross correlation based 3D measurement
    :param volume1:
    :param volume2:
    :param max_shift:
    :return:
    """
    src_ft = np.fft.fftn(volume1)
    target_ft = np.fft.fftn(volume2)
    cross_corr = np.fft.ifftn((src_ft * target_ft.conj()))
    cross_corr_mag = np.abs(np.fft.fftshift(cross_corr))
    search_offset = (np.array(cross_corr.shape) // 2 - max_shift).astype(np.int)
    search_volume = cross_corr_mag[search_offset[0]:search_offset[0] + 2 * max_shift[0],
                    search_offset[1]:search_offset[1] + 2 * max_shift[1],
                    search_offset[2]:search_offset[2] + 2 * max_shift[2]]
    shifts = np.array(np.unravel_index(np.argmax(search_volume), search_volume.shape)).astype(np.int)
    shifts -= np.array(search_volume.shape) // 2
    return shifts

def normalized_x_corr_register_3D(volume1, volume2, max_shift):
    """
    Compute normalized cross correlation based alignment. Takes a really long time but gives better result
    :param volume1:
    :param volume2:
    :param max_shift:
    :return:
    """
    score = np.zeros((2*max_shift[0]+1, 2*max_shift[1]+1, 2*max_shift[2]+1))
    t = np.arange(-max_shift[0], max_shift[0] + 1)
    u = np.arange(-max_shift[1], max_shift[1] + 1)
    v = np.arange(-max_shift[2], max_shift[2] + 1)
    t_indices1 = np.stack((np.max(np.stack((t, np.zeros(t.shape))), axis=0),
               np.min(np.stack((volume1.shape[0] + t, np.ones(t.shape)*volume1.shape[0])), axis=0))).astype(np.int)
    t_indices2 = np.stack((np.max(np.stack((-t, np.zeros(t.shape))), axis=0),
               np.min(np.stack((volume1.shape[0] - t, np.ones(t.shape)*volume1.shape[0])), axis=0))).astype(np.int)
    u_indices1 = np.stack((np.max(np.stack((u, np.zeros(u.shape))), axis=0),
               np.min(np.stack((volume1.shape[1] + u, np.ones(u.shape) * volume1.shape[1])), axis=0))).astype(np.int)
    u_indices2 = np.stack((np.max(np.stack((-u, np.zeros(u.shape))), axis=0),
               np.min(np.stack((volume1.shape[1] - u, np.ones(u.shape) * volume1.shape[1])), axis=0))).astype(np.int)
    v_indices1 = np.stack((np.max(np.stack((v, np.zeros(v.shape))), axis=0),
               np.min(np.stack((volume1.shape[2] + v, np.ones(v.shape) * volume1.shape[2])), axis=0))).astype(np.int)
    v_indices2 = np.stack((np.max(np.stack((-v, np.zeros(v.shape))), axis=0),
               np.min(np.stack((volume1.shape[2] - v, np.ones(v.shape) * volume1.shape[2])), axis=0))).astype(np.int)


    for t_i, ((t11, t12), (t21, t22)) in enumerate(zip(t_indices1.T, t_indices2.T)):
        for u_i, ((u11, u12), (u21, u22)) in enumerate(zip(u_indices1.T, u_indices2.T)):
            for v_i, ((v11, v12), (v21, v22)) in enumerate(zip(v_indices1.T, v_indices2.T)):
                vol1_use = volume1[t11:t12, u11:u12, v11:v12]
                vol2_use = volume2[t21:t22, u21:u22, v21:v22]
                v1_mean_sub = vol1_use - np.mean(vol1_use.astype(np.float))
                v2_mean_sub = vol2_use - np.mean(vol2_use.astype(np.float))
                x_corr = np.sum(v1_mean_sub * v2_mean_sub)
                normalization = np.sqrt(np.sum(v1_mean_sub ** 2) * np.sum(v2_mean_sub ** 2))
                score[t_i, u_i, v_i] = x_corr / normalization

    shifts = np.array(np.unravel_index(np.argmax(score), score.shape)).astype(np.int)
    shifts -= np.array(score.shape) // 2
    return shifts
