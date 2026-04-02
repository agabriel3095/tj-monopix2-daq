# Codex Change Log

Date: 2026-04-01

1. Scan column boundary handling
- Updated the DC/AC electron-conversion selection to treat `stop_column` as exclusive, so `stop_column=448` is interpreted as scanning through real column `447`.
- Files: `tjmonopix2/analysis/plotting.py`, `tjmonopix2/analysis/plotting_noise.py`
- Updated double-column readout masking to respect the exclusive stop boundary while keeping the existing double-column alignment.
- Files: `tjmonopix2/scans/scan_threshold.py`, `tjmonopix2/scans/scan_analog.py`, `tjmonopix2/scans/scan_noise_occupancy.py`, `tjmonopix2/scans/tune_local_threshold.py`, `tjmonopix2/scans/scan_source.py`

Added and removed lines by file:

`tjmonopix2/analysis/plotting.py`
- Added:
  `start_column = self.scan_config['start_column']`
  `stop_column_exclusive = self.scan_config['stop_column']`
  `stop_column_inclusive = stop_column_exclusive - 1`
  `if stop_column_inclusive < start_column:`
  `    stop_column_inclusive = start_column`
  `if start_column < 448 and stop_column_inclusive < 448:`
  `elif start_column >= 448 and stop_column_inclusive < 512:`
- Removed:
  `if self.scan_config['start_column'] in range(0, 448) and self.scan_config['stop_column'] in range(0, 448):`
  `elif self.scan_config['start_column'] in range(448, 512) and self.scan_config['stop_column'] in range(448, 512):`

`tjmonopix2/analysis/plotting_noise.py`
- Added:
  `start_column = self.scan_config['start_column']`
  `stop_column_exclusive = self.scan_config['stop_column']`
  `stop_column_inclusive = stop_column_exclusive - 1`
  `if stop_column_inclusive < start_column:`
  `    stop_column_inclusive = start_column`
  `if start_column < 448 and stop_column_inclusive < 448:`
  `elif start_column >= 448 and stop_column_inclusive < 512:`
- Removed:
  `if self.scan_config['start_column'] in range(0, 448) and self.scan_config['stop_column'] in range(0, 448):`
  `elif self.scan_config['start_column'] in range(448, 512) and self.scan_config['stop_column'] in range(448, 512):`

`tjmonopix2/scans/scan_threshold.py`
- Added:
  `col_disabled += list(range((stop_column + 1) & 0xfffe, 512))`
- Removed:
  `col_disabled += list(range(stop_column + 1, 512))`

`tjmonopix2/scans/scan_analog.py`
- Added:
  `col_disabled += list(range((stop_column + 1) & 0xfffe, 512))`
- Removed:
  `col_disabled += list(range(stop_column + 1, 512))`

`tjmonopix2/scans/scan_noise_occupancy.py`
- Added:
  `col_disabled += list(range((stop_column + 1) & 0xfffe, 512))`
- Removed:
  `col_disabled += list(range(stop_column + 1, 512))`

`tjmonopix2/scans/tune_local_threshold.py`
- Added:
  `col_disabled += list(range((stop_column + 1) & 0xfffe, 512))`
- Removed:
  `col_disabled += list(range(stop_column + 1, 512))`

`tjmonopix2/scans/scan_source.py`
- Added:
  `col_disabled += list(range((stop_column + 1) & 0xfffe, 512))`
- Removed:
  `col_disabled += list(range(stop_column + 1, 512))`

2. Central receiver `DATA_DELAY` configuration
- Added `rx_channels.rx0.DATA_DELAY: 14` to the testbench configuration.
- File: `tjmonopix2/testbench.yaml`
- Added a shared hardware-initialization step that applies receiver register settings from `testbench.yaml` to the DAQ once, before scans run.
- File: `tjmonopix2/system/scan_base.py`
- Removed duplicated per-scan `self.daq.rx_channels['rx0']['DATA_DELAY'] = 14` assignments.
- Files: `tjmonopix2/scans/scan_threshold.py`, `tjmonopix2/scans/scan_analog.py`, `tjmonopix2/scans/scan_noise_occupancy.py`, `tjmonopix2/scans/tune_global_threshold.py`, `tjmonopix2/scans/tune_local_threshold.py`

Added and removed lines by file:

`tjmonopix2/testbench.yaml`
- Added:
  `rx_channels:`
  `  rx0:`
  `    DATA_DELAY: 14`

`tjmonopix2/system/scan_base.py`
- Added:
  `self._configure_rx_channels()`
  `def _configure_rx_channels(self):`
  `rx_config = self.configuration['bench'].get('rx_channels', {})`
  `for receiver, settings in rx_config.items():`
  `if receiver not in self.daq.rx_channels:`
  `self.log.warning("Receiver '%s' configured in testbench, but not present in DAQ", receiver)`
  `for register, value in settings.items():`
  `self.daq.rx_channels[receiver][register] = value`
  `self.log.info("Set %s.%s = %s from testbench config", receiver, register, value)`
- Removed:
  none

`tjmonopix2/scans/scan_threshold.py`
- Removed:
  `self.daq.rx_channels['rx0']['DATA_DELAY'] = 14`

`tjmonopix2/scans/scan_analog.py`
- Removed:
  `self.daq.rx_channels['rx0']['DATA_DELAY'] = 14`

`tjmonopix2/scans/scan_noise_occupancy.py`
- Removed:
  `self.daq.rx_channels['rx0']['DATA_DELAY'] = 14`

`tjmonopix2/scans/tune_global_threshold.py`
- Removed:
  `self.daq.rx_channels['rx0']['DATA_DELAY'] = 14`

`tjmonopix2/scans/tune_local_threshold.py`
- Removed:
  `self.daq.rx_channels['rx0']['DATA_DELAY'] = 14`

3. Warning cleanup
- Fixed the `invalid value encountered in cast` warning by avoiding `NaN` initialization on the unsigned `scan_param_id` field when writing `scan_params`.
- File: `tjmonopix2/system/scan_base.py`
- Fixed the Matplotlib `set_ticklabels()` warning by setting the electron-axis tick positions before assigning labels.
- Files: `tjmonopix2/analysis/plotting.py`, `tjmonopix2/analysis/plotting_noise.py`

Added and removed lines by file:

`tjmonopix2/system/scan_base.py`
- Added:
  `a = np.zeros(shape=(1,), dtype=np.dtype(fields))`
  `a['scan_param_id'] = par_id`
  `for name, dtype in fields[1:]:`
  `if np.issubdtype(dtype, np.floating):`
  `    a[name] = np.nan`
- Removed:
  `a = np.full(shape=(1,), fill_value=np.NaN).astype(np.dtype(fields))`
  `a['scan_param_id'] = par_id` inside the value loop

`tjmonopix2/analysis/plotting.py`
- Added:
  `ax2.set_xticks(ticks)`
  `ax2.set_xticklabels(xticks)`
- Removed:
  none

`tjmonopix2/analysis/plotting_noise.py`
- Added:
  `ax2.set_xticks(ticks)`
  `ax2.set_xticklabels(xticks)`
- Removed:
  none

4. PDF output for threshold tuning scans
- Added analysis and PDF creation for `tune_global_threshold.py` and `tune_local_threshold.py`.
- Files: `tjmonopix2/scans/tune_global_threshold.py`, `tjmonopix2/scans/tune_local_threshold.py`
- Added a dedicated plotting entry point for tuning scans that produces settings, monitoring pages, occupancy/hit summaries, and TDAC map/distribution pages without requiring threshold-fit maps.
- File: `tjmonopix2/analysis/plotting.py`

Added and removed lines by file:

`tjmonopix2/scans/tune_global_threshold.py`
- Added:
  `from tjmonopix2.analysis import analysis, plotting`
  `with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:`
  `a.analyze_data()`
  `if self.configuration['bench']['analysis']['create_pdf']:`
  `with plotting.Plotting(analyzed_data_file=a.analyzed_data_file) as p:`
  `p.create_tuning_plots(include_tdac=True)`
- Removed:
  `pass`

`tjmonopix2/scans/tune_local_threshold.py`
- Added:
  `from tjmonopix2.analysis import analysis, plotting`
  `with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:`
  `a.analyze_data()`
  `if self.configuration['bench']['analysis']['create_pdf']:`
  `with plotting.Plotting(analyzed_data_file=a.analyzed_data_file) as p:`
  `p.create_tuning_plots(include_tdac=True)`
- Removed:
  `pass`

`tjmonopix2/analysis/plotting.py`
- Added:
  `def create_tuning_plots(self, include_tdac=True):`
  `if self.skip_plotting:`
  `    return`
  `self.log.info('Creating tuning plots...')`
  `self.create_parameter_page()`
  `if self._monitoring_enabled_in_pdf():`
  `    self.create_monitoring_summary_table()`
  `    self.create_monitoring_main_page()`
  `self.create_occupancy_map()`
  `self.create_hit_pix_plot()`
  `if include_tdac:`
  `    self.create_tdac_plot()`
  `    self.create_tdac_map()`
  `self.create_tot_plot()`
- Removed:
  none
