def apply_intra_stack_registration(single_channel_stack, registrations, background=0, mode='integer'):
    """
    Apply the computed within z-stack registrations to all channels
    :param stack: dict with channel indices as keys and 3D numpy arrays specific to a single stack in a single channel
    as values
    :param registrations: 2D registration vectors corresponding to each slice
    :return: a list of all channels with a registered stack in each
    """

    if mode == 'float':
        one_channel_registered_stack = np.zeros(single_channel_stack.shape)
        for slice in range(registrations.shape[0]):
            one_channel_registered_stack[slice, ...] = ndi.shift(single_channel_stack[slice],
                                                                 -registrations[slice], cval=background)
            one_channel_registered_stack[one_channel_registered_stack < background] = background
        return one_channel_registered_stack
    else:
        stack_copy = single_channel_stack.copy()
        for slice in range(registrations.shape[0]):
            # need to negate to make it work right
            reg = -np.round(registrations).astype(np.int)[slice]
            reg_slice = stack_copy[slice, ...]
            orig_slice = stack_copy[slice, ...]
            if reg[0] > 0:
                reg_slice = reg_slice[reg[0]:, :]
                orig_slice = orig_slice[:-reg[0], :]
            elif reg[0] < 0:
                reg_slice = reg_slice[:reg[0], :]
                orig_slice = orig_slice[-reg[0]:, :]
            if reg[1] > 0:
                reg_slice = reg_slice[:, reg[1]:]
                orig_slice = orig_slice[:, :-reg[1]]
            elif reg[1] < 0:
                reg_slice = reg_slice[:, :reg[1]]
                orig_slice = orig_slice[:, -reg[1]:]
            reg_slice[:] = orig_slice[:]
        return stack_copy

def exporttiffstack(datacube, name='export'):
    '''
    Save 3D numpy array as a TIFF stack
    :param datacube:
    '''
    if len(datacube.shape) == 2:
        imlist = [Image.fromarray(datacube)]
    else:
        imlist = []
        for i in range(datacube.shape[0]):
            imlist.append(Image.fromarray(datacube[i,...]))
    path = "/Users/henrypinkard/Desktop/{}.tif".format(name)
    imlist[0].save(path, compression="tiff_deflate", save_all=True, append_images=imlist[1:])

def compute_intra_stack_registrations(raw_stacks, nonempty_pixels, max_shift, backgrounds,
            use_channels=[1, 2, 3, 4, 5], sigma_noise=2, abs_reg_bkgd_subtract_sigma=3,
                                       valid_likelihood_threshold=-18):
    """
    Compute registrations for each z slice within a stack using method based on cross correaltion followed by gaussian
    filtering and MLE fitting
    :param channel_stack:
    :param nonempty_pixels:
    :param max_shift:
    :param background:
    :param use_channels:
    :param agreement_k:
    :param likelihood_agreement_threshold:
    :param sigma_noise:
    :return:
    """

    def intra_stack_x_corr_regs(stack, nonempty, max_shift=None):
        """
        Smooth and cross correlate successive slices then take cumulative sum to figure out relative registrations for all
        channels within a stack
        """

        def cross_correlation(src_image, target_image, dont_whiten=True, max_shift=None):
            """
            Compute ND registration between two images
            :param src_image:
            :param target_image:
            :param dont_whiten if, false, normalize before inverse transform (i.e. phase correlation)
            :return:
            """
            src_ft = np.fft.fftn(src_image)
            target_ft = np.fft.fftn(target_image)
            if dont_whiten == True:
                cross_corr = np.fft.ifftn((src_ft * target_ft.conj()))
            else:
                normalized_cross_power_spectrum = (src_ft * target_ft.conj()) / np.abs(src_ft * target_ft.conj())
                normalized_cross_corr = np.fft.ifftn(normalized_cross_power_spectrum)
                cross_corr = normalized_cross_corr
            cross_corr_mag = np.abs(np.fft.fftshift(cross_corr))
            if max_shift == None:
                max_shift = np.min(np.array(cross_corr.shape)) // 2
            search_offset = (np.array(cross_corr.shape) // 2 - int(max_shift)).astype(np.int)
            shifts = np.array(np.unravel_index(np.argmax(
                cross_corr_mag[search_offset[0]:search_offset[0] + 2 * int(max_shift),
                search_offset[1]:search_offset[1] + 2 * int(max_shift)]), (2 * int(max_shift), 2 * int(max_shift))))
            shifts += search_offset
            return shifts.astype(np.float) - np.array(cross_corr.shape) / 2

        def register(current_img, prev_img, max_shift=None, mode='xcorr'):
            """
            gaussian smooth, then compute pairwise registration
            """

            img1 = current_img.astype(np.float)
            img2 = prev_img.astype(np.float)
            if mode == 'xcorr':
                offset = cross_correlation(img1, img2, dont_whiten=True, max_shift=max_shift)
            elif mode == 'phase':
                offset = cross_correlation(img1, img2, dont_whiten=False, max_shift=max_shift)
            return offset

        # compute registrations for each valid set of consecutive slices
        def register_consecutives_slices(z_index, stack, nonempty, channel_index):
            if z_index == 0 or ((not nonempty[z_index - 1]) or (not nonempty[z_index])):
                # take first one as origin and only compute registration if data was acquired at both
                return (0, 0)
            else:
                current_img = stack[channel_index][z_index]
                prev_img = stack[channel_index][z_index - 1]
                return register(current_img, prev_img, max_shift=max_shift, mode='xcorr')

        regs = [[register_consecutives_slices(z_index, stack, nonempty, channel_index) for z_index in
                  range(len(stack[channel_index]))] for  channel_index in range(len(list(stack.keys())))]
        abs_regs = []
        for channel_reg in regs:
            abs_regs.append(np.cumsum(channel_reg, axis=0))
        return abs_regs

    registration_params = []
    for position_index in raw_stacks.keys():
        print('Registering stack slices for position {}'.format(position_index))
        channel_stack = raw_stacks[position_index]

        # make a copy and set background
        channel_stack = {key: channel_stack[key].copy() for key in channel_stack.keys()}
        for channel_index in range(len(channel_stack)):
            channel_stack[channel_index][channel_stack[channel_index] < backgrounds[channel_index]] = backgrounds[channel_index]
        absolute_registrations = intra_stack_x_corr_regs(
                                        channel_stack, nonempty_pixels[position_index], max_shift=max_shift)
        zero_centered_regs = []
        background_shifts = []
        for channel_reg in absolute_registrations:
            background_shift = np.array([
                    ndi.filters.gaussian_filter1d(channel_reg[:, 0], sigma=abs_reg_bkgd_subtract_sigma),
                    ndi.filters.gaussian_filter1d(channel_reg[:, 1], sigma=abs_reg_bkgd_subtract_sigma)]).T
            background_shifts.append(background_shift)
            zero_centered_regs.append(channel_reg - background_shift)
        # plt.figure(); plt.plot(np.array(absolute_registrations)[1:,:, 0].T, '.-')
        # plt.xlabel('z-slice index'); plt.ylabel('x shift (pixels)'); plt.legend(['Channel {}'.format(i) for i in range(1,6)])
        # plt.plot(np.array(background_shifts)[1:,:, 0].T, 'k--');
        #
        # plt.figure(); plt.plot(np.array(zero_centered_regs)[1:,:, 0].T,'.-'); plt.ylim([-24, 24]);
        # plt.legend(['Channel {}'.format(i) for i in range(1,6)]);
        # plt.xlabel('z-slice index'); plt.ylabel('x shift (pixels)'); plt.show()

        #ignore empty slices
        regs_to_use = np.array([zero_centered_regs[channel]
                                for channel in use_channels])[..., nonempty_pixels[position_index], :]
        # plt.figure(); plt.plot(regs_to_use[:,:, 0].T); plt.ylim([-24, 24]); plt.legend([str(i) for i in range(5)])

        ####### compute per-chanel likelihoods #########
        x = np.linspace(-max_shift, max_shift, 500)
        #(channels) x (z slice) x (2 dims of registration) x (parameter space)
        likelihoods = np.zeros((regs_to_use.shape[:2]) + (2, x.size))
        for channel_index in range(regs_to_use.shape[0]):
            for z_index in range(regs_to_use.shape[1]):
                registration = regs_to_use[channel_index, z_index]
                likelihoods[channel_index, z_index, :, :] = (1 / (np.sqrt(2*np.pi) * sigma_noise) *
                    np.exp(-((np.stack(2*[x]) - np.expand_dims(registration, axis=1)) ** 2) / (2 * sigma_noise ** 2)))

        ##### Compute MLE over only the best slices
        mles_all_channels = np.zeros(likelihoods.shape[1:3])
        mls_all_channels = np.zeros(likelihoods.shape[1:3])
        for z_index in range(likelihoods.shape[1]):
            likelihood_prod_all_channels = np.prod(likelihoods[:, z_index, :, :], axis=0)
            # plt.figure();  plt.semilogy(likelihood_prod_all_channels[0], '.-')
            #MLE
            mles_all_channels[z_index] = x[np.argmax(likelihood_prod_all_channels, axis=1)]
            mls_all_channels[z_index] = np.max(likelihood_prod_all_channels, axis=1)
        composite_log_likelihood = np.log(np.prod(mls_all_channels, axis=1))
        # smoothed_log_likeliood = ndi.gaussian_filter1d(composite_log_likelihood, likelihood_threshold_smooth_sigma)
        # plt.figure(); plt.plot(composite_log_likelihood, '.-'); plt.ylabel('log(likelihood)'); plt.xlabel('z-slice index'); plt.show()
        # plt.plot(smoothed_log_likeliood, '.-')
        valid_mles = composite_log_likelihood > valid_likelihood_threshold

        # mles_all_channels[np.logical_not(valid_mles)] = sin_pred_movement[np.logical_not(valid_mles)]
        mles_all_channels[np.logical_not(valid_mles)] = 0
        # plt.figure(); plt.plot(mles_all_channels)

        #apply the computed registrations to slices that have pixel data
        registrations = np.zeros(absolute_registrations[0].shape)
        registrations[nonempty_pixels[position_index]] = mles_all_channels

        # for channel in range(6):
        #     registered_stack = apply_intra_stack_registration(channel_stack[channel], registrations)
        #     exporttiffstack(registered_stack, 'registered channel {}'.format(channel))
        #     exporttiffstack(channel_stack[channel], 'unregistered channel {}'.format(channel))
        registration_params.append(registrations)
    return registration_params
