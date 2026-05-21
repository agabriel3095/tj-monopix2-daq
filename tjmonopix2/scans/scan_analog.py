#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

from tjmonopix2.analysis import analysis, plotting
from tjmonopix2.scans.shift_and_inject import (get_scan_loop_mask_steps,
                                               shift_and_inject,
                                               DEFAULT_PULSE_START_CNFG)
from tjmonopix2.system.scan_base import ScanBase
from tqdm import tqdm

import yaml, json, argparse

scan_configuration = {
    'start_column': 288,
    'stop_column': 290,
    'start_row': 0,
    'stop_row': 512,
    # Analog scans keep the same configurable pulse timing interface as the
    # threshold scans so BCID studies can reuse this scan path if needed.
    'pulse_start_cnfg': DEFAULT_PULSE_START_CNFG,
}


class AnalogScan(ScanBase):
    scan_id = 'analog_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][start_column:stop_column, start_row:stop_row] = True


        #Feature to call the defaukt registers for chip and FE, saved on a json file. 
        # ATTENTION!! all hardcoded wite.registers MUST go below these lines
        def_regs = self.get_config_param("def_regs")
        regs_json = self.get_config_param("regs_json", "../chip_registers.json")
        chip = self.get_config_param("chip")
        fe = self.get_config_param("fe")
        if def_regs:
            self.load_regs_config(json_path=regs_json, chip=chip, fe=fe)


        # self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100
        # self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 4# TDAC=4 (default)
        # self.chip.masks['hitor'][0, 0] = True

        col_bad = []
        # W8R6 bad columns (246 to 251 included: double-cols will be disabled)
        col_bad += [248]
        col_bad += [436]
        # W8R13 pixels that fire even when disabled
        col_bad += list(range(383,415)) # chip w8r13
        col_bad += list(range(0,40)) # chip w8r13
        col_bad += list(range(448,512)) # HV col disabled
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
            #print(f"test i {enumerate(reg_values)}")
            # EN_RO_CONFsource /home/labb2/tj-monopix2-daq-development/venv/bin/activate
            self.chip._write_register(155+i, v)
            # EN_BCID_CONF (to disable BCID distribution on cols under test, use 0 instead of v, doing this the TOT is 0 since Le and trailing edge are not assigned BCID is missing)
            # To enable it all the matrix (higher I_LV and Temp), use  self.chip._write_register(171+i, 0xffff)
            # To enable only the used columns, use  self.chip._write_register(171+i, v)
            # To disable BCID distribution in all columns, use  self.chip._write_register(171+i, 0)
            # self.chip._write_register(171+i, v)
            self.chip._write_register(171+i, 0xffff)
            # self.chip._write_register(171+i, 0)
            # EN_RO_RST_CONF
            self.chip._write_register(187+i, v)
            # EN_FREEZE_CONF
            self.chip._write_register(203+i, v)
            # Read back
            # print(f"{i:3d} {v:016b} {self.chip._get_register_value(155+i):016b} {self.chip._get_register_value(171+i):016b} {self.chip._get_register_value(187+i):016b} {self.chip._get_register_value(203+i):016b}")

        # self.chip.registers["ITHR"].write(45)

        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)

        # self.chip.registers["ITHR"].write(50)
        # self.chip.registers["IDB"].write(100)

        self.chip.registers["VL"].write(30)
        self.chip.registers["VH"].write(150)
        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)


        # self.chip.registers["FREEZE_START_CONF"].write(250)
        # self.chip.registers["READ_START_CONF"].write(253)
        # self.chip.registers["READ_STOP_CONF"].write(255)
        # self.chip.registers["LOAD_CONF"].write(270)
        # self.chip.registers["FREEZE_STOP_CONF"].write(271)
        # self.chip.registers["STOP_CONF"].write(271)


    def _scan(self, n_injections=100, pulse_start_cnfg=DEFAULT_PULSE_START_CNFG, **_):
        pbar = tqdm(total=get_scan_loop_mask_steps(self.chip), unit='Mask steps')
        with self.readout(scan_param_id=0):
            shift_and_inject(chip=self.chip, n_injections=n_injections, pbar=pbar, scan_param_id=0,
                             PulseStartCnfg=pulse_start_cnfg, step_callback=lambda: self.update_readout_progress(pbar))
        self.update_readout_progress(pbar)
        pbar.close()

        self.log.success('Scan finished')

    def _analyze(self):
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:
            a.analyze_data()

        if self.configuration['bench']['analysis']['create_pdf']:
            with plotting.Plotting(analyzed_data_file=a.analyzed_data_file) as p:
                p.create_standard_plots()


# if __name__ == "__main__":
#     with AnalogScan(scan_config=scan_configuration) as scan:
#         scan.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--def_regs", help="Force the use of default registers", action="store_true")
    parser.add_argument("--regs_json", type=str, help="Path to JSON file with regs configs", default="../chip_registers.json")
    parser.add_argument("--chip", type=str, help="Chip name (e.g., W8R6)")
    parser.add_argument("--fe", type=str, help="FE name (e.g., HVC or DCC)")
    parser.add_argument("--pulse-start-cnfg", type=int, default=scan_configuration.get("pulse_start_cnfg", DEFAULT_PULSE_START_CNFG))
    parser.add_argument("--h5_config_file", type=str, default=None)
    args = parser.parse_args()

    # Mirror the CLI-selected pulse start into the saved scan configuration.
    scan_configuration["pulse_start_cnfg"] = args.pulse_start_cnfg

    if args.h5_config_file:
        scan_configuration["chip_config_file"] = args.h5_config_file

    with AnalogScan(scan_config=scan_configuration) as scan:
        if args.def_regs:
            scan.configuration.setdefault("configure", {})
            scan.configuration["configure"].update({
                "def_regs": args.def_regs,
                "regs_json": args.regs_json,
                "chip": args.chip,
                "fe": args.fe
            })
        scan.start()
