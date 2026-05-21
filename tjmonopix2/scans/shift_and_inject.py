#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

'''
    This module takes care of the mask shifting and injection of the supported chips
    in order to keep actual scans cleaner.
'''


# Keep the historical pulse-start setting as the shared default so scans only
# need to override it when they are explicitly studying BCID timing.
DEFAULT_PULSE_START_CNFG = 19


def shift_and_inject(chip, n_injections, pbar=None, scan_param_id=0, masks=['injection', 'enable'], pattern='default', cache=False, skip_empty=True, PulseStartCnfg=DEFAULT_PULSE_START_CNFG, wait_cycles=1, latency=1400, step_callback=None, progress_callback=None):
    ''' Regular mask shift and analog injection function.

    Parameters:
    ----------
        chip : chip object
            Chip object
        n_injections : int
            Number of injections per loop.
        pbar : tqdm progressbar
            Tqdm progressbar
        scan_param_id : int
            Scan parameter id of actual scan loop
        masks : list
            List of masks ('injection', 'enable', 'hitbus') which should be shifted during scan loop.
        pattern : string
            Injection pattern ('default', 'hitbus', ...)
        cache : boolean
            If True use mask caching for speedup. Default is False.
        skip_empty : boolean
            If True skip empty mask steps for speedup. Default is True.
    '''
    callback = step_callback if step_callback is not None else progress_callback

    for fe, active_pixels in chip.masks.shift(masks=masks, pattern=pattern, cache=cache, skip_empty=skip_empty):
        if not fe == 'skipped':
            # Derive PulseStopCnfg from the selected start so all scans keep the
            # same injection width while still allowing BCID phase shifts.
            chip.inject(PulseStartCnfg=PulseStartCnfg, PulseStopCnfg=PulseStartCnfg + 900, repetitions=n_injections, wait_cycles=wait_cycles, latency=latency)
            # chip.inject(PulseStartCnfg=PulseStartCnfg, PulseStopCnfg=PulseStartCnfg + 512, repetitions=n_injections, wait_cycles=wait_cycles, latency=latency) --> default in dev branch
        if pbar is not None:
            pbar.update(1)
            if callback is not None:
                callback()


def get_scan_loop_mask_steps(chip, pattern='default'):
    ''' Returns total number of mask steps for specific pattern

    Parameters:
    ----------
        chip : chip object
            Chip object
        pattern : string
            Injection pattern ('default', 'hitbus', ...)
    '''

    return chip.masks.get_mask_steps(pattern=pattern)
