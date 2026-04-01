try:
    from tjmonopix2.scans.tot_calibration_common import run_tot_calibration
except ImportError:
    from tot_calibration_common import run_tot_calibration


if __name__ == '__main__':
    run_tot_calibration('fit3par')
