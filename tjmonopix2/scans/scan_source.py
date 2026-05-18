#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import time
import threading
from tqdm import tqdm

import yaml, json, argparse

from tjmonopix2.analysis import analysis, plotting
from tjmonopix2.system.scan_base import ScanBase

scan_configuration = {
    'start_column': 224,
    'stop_column': 448,
    # 'start_column': 288,
    # 'stop_column': 290,    
    'start_row': 0,
    'stop_row': 512,

    'scan_timeout': 300,    # Timeout for scan after which the scan will be stopped, in seconds; if False no limit on scan time

    'tot_calib_file': None#'output_data/module_0/chip_0/20240806_121701_threshold_scan_interpreted.h5'    # path to ToT calibration file for charge to e⁻ conversion, if None no conversion will be done

}


class SourceScan(ScanBase):
    scan_id = 'source_scan'

    stop_scan = threading.Event()

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True

        #Feature to call the defaukt registers for chip and FE, saved on a json file. 
        # ATTENTION!! all hardcoded wite.registers MUST go below these lines
        # Example  # self.chip.registers["ITHR"].write(50)
        def_regs = self.get_config_param("def_regs")
        regs_json = self.get_config_param("regs_json", "../chip_registers.json")
        chip = self.get_config_param("chip")
        fe = self.get_config_param("fe")
        if def_regs:
            self.load_regs_config(json_path=regs_json, chip=chip, fe=fe)

        # TDAC=4 for threshold tuning 0b100
        # self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 4 # TDAC=4 (default)

        # Read masked pixels from masked_pixels.yaml
        with open("output_data/module_0/chip_0/masked_pixels.yaml") as f:
            masked_pixels = yaml.full_load(f)

        for i in range(0, len(masked_pixels['masked_pixels'])):
            row = masked_pixels['masked_pixels'][i]['row']
            col = masked_pixels['masked_pixels'][i]['col']
            self.chip.masks.disable_mask[col, row] = False
            # self.chip.masks['tdac'][col, row] = 0 # --> Max solution to disable the pixel BUT not store in use_pixel NOR in masks.enable


        # Disable W8R13 bad/broken columns (25, 160, 161, 224, 274, 383-414 included, 447) and pixels
        #self.chip.masks['enable'][25,:] = False  # Many pixels don't fire
        #self.chip.masks['enable'][160:162,:] = False  # Wrong/random ToT
        # self.chip.masks['enable'][224,:] = False  # Many pixels don't fire
        #self.chip.masks['enable'][274,:] = False  # Many pixels don't fire
        #self.chip.masks['enable'][383:415,:] = False  # Wrong/random ToT
        #self.chip.masks['enable'][447,:] = False  # Many pixels don't fire
        # self.chip.masks['enable'][75,159] = False
        # self.chip.masks['enable'][163,219] = False
        # self.chip.masks['enable'][427,259] = False
        # self.chip.masks['enable'][219,161] = False # disab20230620_153108_threshold_scanle hottest pixel on chip
        # self.chip.masks['enable'][214,88] = False
        # self.chip.masks['enable'][215,101] = False
        # self.chip.masks['enable'][191:223,:] = False  # cols 191-223 are broken since Nov/dec very low THR

        # For TJ-MP2 training 13 Apr 26
        # chip w8r13 bad cols
        #Disable W8R13 bad/broken columns (25, 160, 161, 224, 274, 383-414 included, 447) and pixels
        self.chip.masks['enable'][25,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][160:162,:] = False  # Wrong/random ToT
        self.chip.masks['enable'][224,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][274,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][383:415,:] = False  # Wrong/random ToT
        self.chip.masks['enable'][447,:] = False  # Many pixels don't fire


        # Disable W8R13 bad/broken columns (25, 160, 161, 224, 274, 383-414 included, 447) and pixels
        self.chip.masks['enable'][25,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][160:162,:] = False  # Wrong/random ToT
        self.chip.masks['enable'][224,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][274,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][383:415,:] = False  # Wrong/random ToT
        self.chip.masks['enable'][447,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][450,68] = False
        self.chip.masks['enable'][288,316] = False
        self.chip.masks['enable'][163,219] = False
        self.chip.masks['enable'][427,259] = False
        self.chip.masks['enable'][219,161] = False # disab20230620_153108_threshold_scanle hottest pixel on chip
        self.chip.masks['enable'][214,88] = False
        self.chip.masks['enable'][215,101] = False
        self.chip.masks['enable'][450,463] = False
        self.chip.masks['enable'][191:223,:] = False  # cols 191-223 are broken since Nov/dec very low THR



        col_bad = [] #
        # W8R6 bad columns (246 to 251 included: double-cols will be disabled)
        # col_bad = [248]
        # Disable readout for double-columns of col_disabled and those outside start_column:stop_column
        col_disabled = col_bad
        col_disabled += list(range(0, start_column & 0xfffe))
        col_disabled += list(range((stop_column + 1) & 0xfffe, 512))
        reg_values = [0xffff] * 16
        for col in col_disabled:
            dcol = col // 2
            reg_values[dcol//16] &= ~(1 << (dcol % 16))
        # print(" ".join(f"{x:016b}" for x in reg_values))
        for i, v in enumerate(reg_values):
            # EN_RO_CONF
            self.chip._write_register(155+i, v)
            # EN_BCID_CONF (to disable BCID distribution on cols under test, use 0 instead of v, doing this the TOT is 0 since Le and trailing edge are not assigned BCID is missing)
            # if below is commented the BCID seems to be enabled in all the matrix higher I_LV and Temp
            # To enable it all the matrix (higher I_LV and Temp), use  self.chip._write_register(171+i, 0xffff)
            # To enable only the used columns, use  self.chip._write_register(171+i, v)
            # To disable BCID distribution in all columns, use  self.chip._write_register(171+i, 0)
            self.chip._write_register(171+i, 0xffff)
            #self.chip._write_register(171+i, 0)
            #self.chip._write_register(171+i, v)
            # EN_RO_RST_CONF
            self.chip._write_register(187+i, v)
            # EN_FREEZE_CONF
            self.chip._write_register(203+i, v)
            # Read back
            # print(f"{i:3d} {v:016b} {self.chip._get_register_value(155+i):016b} {self.chip._get_register_value(171+i):016b} {self.chip._get_register_value(187+i):016b} {self.chip._get_register_value(203+i):016b}")

        # self.chip.registers["ITHR"].write(50)


        self.chip.masks.apply_disable_mask()
        self.chip.masks.update()

        


    def _scan(self, scan_timeout=10, **_):
        def timed_out():
            if scan_timeout:
                current_time = time.time()
                if current_time - start_time > scan_timeout:
                    self.log.info('Scan timeout was reached')
                    return True
            return False

        self.pbar = tqdm(total=scan_timeout, unit='')  # [s]
        start_time = time.time()

        with self.readout():
            self.stop_scan.clear()

            while not (self.stop_scan.is_set() or timed_out()):
                try:
                    time.sleep(1)

                    # Update progress bar
                    try:
                        self.pbar.update(1)
                        self.update_readout_progress(self.pbar)
                    except ValueError:
                        pass

                except KeyboardInterrupt:  # React on keyboard interupt
                    self.stop_scan.set()
                    self.log.info('Scan was stopped due to keyboard interrupt')

        self.update_readout_progress(self.pbar)
        self.pbar.close()
        self.log.success('Scan finished')

    def _analyze(self):
        tot_calib_file = self.configuration['scan'].get('tot_calib_file', None)
        if tot_calib_file is not None:
            self.configuration['bench']['analysis']['cluster_hits'] = True

        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', tot_calib_file=tot_calib_file, **self.configuration['bench']['analysis']) as a:
            a.analyze_data()

        if self.configuration['bench']['analysis']['create_pdf']:
            with plotting.Plotting(analyzed_data_file=a.analyzed_data_file) as p:
                p.create_standard_plots()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--def_regs", help="Force the use of default registers", action="store_true")
    parser.add_argument("--regs_json", type=str, help="Path to JSON file with regs configs", default="../chip_registers.json")
    parser.add_argument("--chip", type=str, help="Chip name (e.g., W8R6)")
    parser.add_argument("--fe", type=str, help="FE name (e.g., HVC or DCC)")
    parser.add_argument("--h5_config_file", type=str, default=None)
    args = parser.parse_args()

    if args.h5_config_file:
        scan_configuration["chip_config_file"] = args.h5_config_file

    with SourceScan(scan_config=scan_configuration) as scan:
        if args.def_regs:
            scan.configuration.setdefault("configure", {})
            scan.configuration["configure"].update({
                "def_regs": args.def_regs,
                "regs_json": args.regs_json,
                "chip": args.chip,
                "fe": args.fe
            })
        scan.start()
