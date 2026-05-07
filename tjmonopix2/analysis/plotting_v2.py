"""Offline-friendly plotting module with selectable plots and axis overrides."""

import os
import copy
import shutil
import math
import tables as tb
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json

from collections import OrderedDict
from scipy.optimize import curve_fit
from matplotlib.figure import Figure
from matplotlib.artist import setp
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors, cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FormatStrFormatter, ScalarFormatter
from tqdm.auto import tqdm

from tjmonopix2.system import logger
from tjmonopix2.analysis import analysis_utils as au
from tjmonopix2.analysis import monitoring

from tjmonopix2.system import telegram_bot

TITLE_COLOR = '#07529a'
OVERTEXT_COLOR = '#07529a'

SCURVE_CHI2_UPPER_LIMIT = 50

DACS = {'TJMONOPIX2': ['IBIAS', 'ITHR',
                       'ICASN', 'IDB',
                       'ITUNE', 'ICOMP',
                       'IDEL', 'IRAM',
                       'VRESET', 'VCASP',
                       'VCASC', 'VCLIP', ]
        }

SOURCE_LIKE_SCANS = {'source_scan', 'ext_trigger_scan', 'noise_occupancy_scan'}
TOT_LIKE_SCANS = {'analog_scan', 'threshold_scan', 'global_threshold_tuning', 'source_scan', 'ext_trigger_scan', 'calibrate_tot', 'noise_occupancy_scan'}
THRESHOLD_LIKE_SCANS = {'threshold_scan', 'calibrate_tot'}


class Plotting(object):
    """Create plots from an interpreted HDF5 file.

    Compared to the legacy plotting module, this version adds:
    - named plot selection
    - optional axis-range overrides
    - optional progress display
    - optional Telegram notifications
    """

    def __init__(self, analyzed_data_file, pdf_file=None, level='preliminary', mask_noisy_pixels=False, internal=False, save_single_pdf=False, save_png=False, notify=False, show_progress=True, axis_ranges=None, create_output=True, map_split_config=None):
        self.log = logger.setup_derived_logger('Plotting')

        self.plot_cnt = 0
        self.save_single_pdf = save_single_pdf
        self.save_png = save_png
        self.level = level
        self.mask_noisy_pixels = mask_noisy_pixels
        self.internal = internal
        self.clustered = False
        self.skip_plotting = False
        self.cb_side = False
        self._module_type = None
        self.notify = notify
        self.show_progress = show_progress
        self.axis_ranges = axis_ranges or {}
        self.create_output = create_output
        self._active_plot_id = None
        self.map_split_config = map_split_config or {}
        self._scan_area_tile_specs_cache = {}

        if pdf_file is None:
            self.filename = '.'.join(
                analyzed_data_file.split('.')[:-1]) + '.pdf'
        else:
            self.filename = pdf_file
        self.out_file = PdfPages(self.filename) if self.create_output else None

        try:
            if isinstance(analyzed_data_file, str):
                self.in_file = tb.open_file(analyzed_data_file, 'r')
                root = self.in_file.root
            else:
                self.in_file = None
                root = analyzed_data_file
        except IOError:
            self.log.warning('Interpreted data file does not exist!')
            self.skip_plotting = True
            return

        self.scan_config = au.ConfigDict(root.configuration_in.scan.scan_config[:])
        self.run_config = au.ConfigDict(root.configuration_in.scan.run_config[:])
        self.chip_settings = au.ConfigDict(root.configuration_in.chip.settings[:])
        self.cols = 512
        self.rows = 512
        self.num_pix = self.rows * self.cols
        self.plot_box_bounds = [0.5, self.cols + 0.5, self.rows + 0.5, 0.5]
        try:
            self.scan_params = root.configuration_in.scan.scan_params[:]
        except tb.NoSuchNodeError:
            self.scan_params = None

        self.registers = au.ConfigDict(root.configuration_in.chip.registers[:])

        if self.run_config['scan_id']:  # TODO: define 'usual' scans
            self.enable_mask = self._mask_disabled_pixels(root.configuration_in.chip.use_pixel[:], self.scan_config)
            self.n_enabled_pixels = len(self.enable_mask[~self.enable_mask])
            self.tdac_mask = root.configuration_in.chip.masks.tdac[:]

        # self.calibration = {e[0].decode('utf-8'): float(e[1].decode('utf-8')) for e in root.configuration_in.chip.calibration[:]}

        if 'VCAL_LOW' in self.scan_params.dtype.names:
            self.scan_parameter_range = np.array(self.scan_params['VCAL_HIGH'] - self.scan_params['VCAL_LOW'], dtype=float)
        elif 'VCAL_LOW_start' in self.scan_config:
            self.scan_parameter_range = [self.scan_config['VCAL_HIGH'] - v for v in
                                         range(self.scan_config['VCAL_LOW_start'],
                                               self.scan_config['VCAL_LOW_stop'] - 1,
                                               self.scan_config['VCAL_LOW_step'])]
        elif 'VTH_start' in self.scan_config:
            self.scan_parameter_range = list(range(self.scan_config['VTH_start'],
                                                   self.scan_config['VTH_stop'],
                                                   -1 * self.scan_config['VTH_step']))
        else:
            self.scan_parameter_range = None

        try:
            self.HistTdcStatus = root.HistTdcStatus[:]
        except tb.NoSuchNodeError:
            self.HistTdcStatus = None
        self.HistOcc = root.HistOcc[:]
        self.HistTot = root.HistTot
        if self.run_config['scan_id'] in ['threshold_scan', 'calibrate_tot', 'fast_threshold_scan', 'in_time_threshold_scan', 'autorange_threshold_scan', 'crosstalk_scan']:
            self.ThresholdMap = root.ThresholdMap[:, :]
            self.Chi2Map = root.Chi2Map[:, :]
            self.Chi2Sel = (self.Chi2Map > 0) & (self.Chi2Map < SCURVE_CHI2_UPPER_LIMIT) & (~self.enable_mask)
            self.n_failed_scurves = self.n_enabled_pixels - len(self.Chi2Map[self.Chi2Sel])
            self.NoiseMap = root.NoiseMap[:]

        if self.mask_noisy_pixels:
            noisy_pixels = np.where(self.HistOcc > self.mask_noisy_pixels)
            for i in range(len(noisy_pixels[0])):
                self.log.warning('Disabling noisy pixel ({0}, {1})'.format(noisy_pixels[0][i], noisy_pixels[1][i]))
                self.enable_mask[noisy_pixels[0][i], noisy_pixels[1][i]] = True
            self.n_enabled_pixels = len(self.enable_mask[~self.enable_mask])
            self.log.warning('Disabled {} noisy pixels in total.'.format(len(noisy_pixels[0])))

        try:
            _ = root.Cluster
            self.HistClusterSize = root.HistClusterSize[:]
            self.HistClusterShape = root.HistClusterShape[:]
            self.HistClusterTot = root.HistClusterTot[:]
            self.clustered = True
        except tb.NoSuchNodeError:
            pass

        try:
            conversion_factors = au.ConfigDict(root.configuration_in.bench.electron_conversion[:])
            start_column = self.scan_config['start_column']
            stop_column_exclusive = self.scan_config['stop_column']
            stop_column_inclusive = stop_column_exclusive - 1
            if stop_column_inclusive < start_column:
                stop_column_inclusive = start_column

            if start_column < 448 and stop_column_inclusive < 448:  # TODO: Get rid of hardcoded values.
                self.electron_conversion = conversion_factors['DC_coupled']
            elif start_column >= 448 and stop_column_inclusive < 512:
                self.electron_conversion = conversion_factors['AC_coupled']
            else:
                self.log.warning('Both DC and AC coupled pixels enabled. Choose ambiguous electron conversion factor of DC falvors!')
                self.electron_conversion = conversion_factors['DC_coupled']
            self.plot_electron_axis = True
        except tb.NoSuchNodeError:
            self.plot_electron_axis = False

        # Monitoring data
        self.monitoring_cfg = None
        self.monitoring_group = None
        self.monitoring_summary = None
        self.monitoring_tables = {}
        self.monitoring_scan_start = None
        self.monitoring_scan_stop = None
        try:
            self.monitoring_cfg = monitoring.load_monitoring_config_from_root(root)
        except Exception:
            self.monitoring_cfg = None
        try:
            self.monitoring_group = root.monitoring
            self.monitoring_scan_start = getattr(self.monitoring_group._v_attrs, 'scan_start', None)
            self.monitoring_scan_stop = getattr(self.monitoring_group._v_attrs, 'scan_stop', None)
            if hasattr(self.monitoring_group, 'summary'):
                self.monitoring_summary = self.monitoring_group.summary[:]
            if hasattr(self.monitoring_group, 'power'):
                self.monitoring_tables['power'] = self._snapshot_monitoring_table(self.monitoring_group.power)
            if hasattr(self.monitoring_group, 'env'):
                self.monitoring_tables['env'] = self._snapshot_monitoring_table(self.monitoring_group.env)
        except Exception:
            self.monitoring_group = None

        try:
            in_file.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.out_file is not None and isinstance(self.out_file, PdfPages):
            self.log.info('Closing output PDF file: {0}'.format(self.out_file._file.fh.name))
            self.out_file.close()
            shutil.copyfile(self.filename, os.path.join(os.path.split(self.filename)[0], 'last_scan.pdf'))
        if getattr(self, 'in_file', None) is not None:
            self.in_file.close()
        if self.notify:
            telegram_bot.send_message_scan(self.run_config, self.filename)

    def _sum_hist_tot(self):
        '''Return the full ToT histogram without loading HistTot completely.'''
        if isinstance(self.HistTot, np.ndarray):
            return self.HistTot.sum(axis=(0, 1, 2)).T

        hist = np.zeros(self.HistTot.shape[3], dtype=np.uint64)
        for scan_param_id in range(self.HistTot.shape[2]):
            for col_start in range(0, self.HistTot.shape[0], 64):
                col_stop = min(col_start + 64, self.HistTot.shape[0])
                hist += self.HistTot[col_start:col_stop, :, scan_param_id, :].sum(axis=(0, 1)).astype(np.uint64)
        return hist

    def _sum_hist_tot_by_scan_param(self):
        '''Return ToT-vs-scan-parameter histogram without loading HistTot fully.'''
        if isinstance(self.HistTot, np.ndarray):
            return self.HistTot.sum(axis=(0, 1)).T

        hist = np.zeros((self.HistTot.shape[3], self.HistTot.shape[2]), dtype=np.uint64)
        for scan_param_id in range(self.HistTot.shape[2]):
            for col_start in range(0, self.HistTot.shape[0], 64):
                col_stop = min(col_start + 64, self.HistTot.shape[0])
                hist[:, scan_param_id] += self.HistTot[col_start:col_stop, :, scan_param_id, :].sum(axis=(0, 1)).astype(np.uint64)
        return hist

    def _sum_hist_tot_per_pixel(self):
        '''Return per-pixel ToT counts and weighted sums without loading HistTot fully.'''
        bins = np.arange(self.HistTot.shape[3], dtype=np.uint64)
        if isinstance(self.HistTot, np.ndarray):
            counts = self.HistTot.sum(axis=2).astype(np.uint64)
            total = counts.sum(axis=2)
            weighted = np.tensordot(counts, bins, axes=([2], [0]))
            return total, weighted

        total = np.zeros(self.HistTot.shape[:2], dtype=np.uint64)
        weighted = np.zeros(self.HistTot.shape[:2], dtype=np.uint64)
        for scan_param_id in range(self.HistTot.shape[2]):
            for col_start in range(0, self.HistTot.shape[0], 64):
                col_stop = min(col_start + 64, self.HistTot.shape[0])
                chunk = self.HistTot[col_start:col_stop, :, scan_param_id, :].astype(np.uint64)
                total[col_start:col_stop, :] += chunk.sum(axis=2)
                weighted[col_start:col_stop, :] += np.tensordot(chunk, bins, axes=([2], [0]))
        return total, weighted

    def _plot_scan_area_map(self, hist, suffix, title, **kwargs):
        '''Plot a cropped scan-area copy with original row/column coordinates.'''
        if not all(k in self.scan_config for k in ('start_column', 'stop_column', 'start_row', 'stop_row')):
            return

        start_column = int(self.scan_config['start_column'])
        stop_column = int(self.scan_config['stop_column'])
        start_row = int(self.scan_config['start_row'])
        stop_row = int(self.scan_config['stop_row'])
        hist_scan_area = hist[start_row:stop_row, start_column:stop_column]
        old_plot_box_bounds = self.plot_box_bounds
        self.plot_box_bounds = [
            start_column + 0.5,
            stop_column + 0.5,
            stop_row + 0.5,
            start_row + 0.5,
        ]
        try:
            self._plot_occupancy(hist=hist_scan_area,
                                 suffix=suffix,
                                 title=title,
                                 aspect='auto',
                                 **kwargs)
        finally:
            self.plot_box_bounds = old_plot_box_bounds

    def _plot_scan_area_map_tiles(self, hist, suffix, title, plot_kind='occupancy', **kwargs):
        """Plot optional zoom tiles inside the scanned area for 2D maps."""
        tile_specs = self._get_scan_area_tile_specs()
        if not tile_specs:
            return

        renderer = self._plot_occupancy if plot_kind == 'occupancy' else self._plot_fancy_occupancy
        old_plot_box_bounds = self.plot_box_bounds
        try:
            for tile in tile_specs:
                hist_tile = hist[tile['row_start']:tile['row_stop'], tile['col_start']:tile['col_stop']]
                self.plot_box_bounds = [
                    tile['col_start'] + 0.5,
                    tile['col_stop'] + 0.5,
                    tile['row_stop'] + 0.5,
                    tile['row_start'] + 0.5,
                ]
                renderer(
                    hist=hist_tile,
                    suffix=f"{suffix}_r{tile['row_index'] + 1:02d}_c{tile['col_index'] + 1:02d}",
                    title=f"{title} zoom r{tile['row_index'] + 1}/{tile['row_splits']} c{tile['col_index'] + 1}/{tile['col_splits']}",
                    **kwargs,
                )
        finally:
            self.plot_box_bounds = old_plot_box_bounds

    def _plot_scan_area_fancy_map(self, hist, suffix, title, **kwargs):
        '''Plot a cropped scan-area copy for maps with projections.'''
        if not all(k in self.scan_config for k in ('start_column', 'stop_column', 'start_row', 'stop_row')):
            return

        start_column = int(self.scan_config['start_column'])
        stop_column = int(self.scan_config['stop_column'])
        start_row = int(self.scan_config['start_row'])
        stop_row = int(self.scan_config['stop_row'])
        hist_scan_area = hist[start_row:stop_row, start_column:stop_column]
        old_plot_box_bounds = self.plot_box_bounds
        self.plot_box_bounds = [
            start_column + 0.5,
            stop_column + 0.5,
            stop_row + 0.5,
            start_row + 0.5,
        ]
        try:
            self._plot_fancy_occupancy(hist=hist_scan_area,
                                       suffix=suffix,
                                       title=title,
                                       **kwargs)
        finally:
            self.plot_box_bounds = old_plot_box_bounds

    def _get_current_map_split_config(self):
        """Return the split configuration for the active plot, if any."""
        if not self.map_split_config or self._active_plot_id is None:
            return None
        config = self.map_split_config.get(self._active_plot_id)
        if not config:
            return None
        return config

    def _get_scan_area_tile_specs(self):
        """Return valid scan-area tile definitions for optional zoomed map plots."""
        split_config = self._get_current_map_split_config()
        cache_key = self._active_plot_id
        if cache_key in self._scan_area_tile_specs_cache:
            return self._scan_area_tile_specs_cache[cache_key]
        if not split_config:
            self._scan_area_tile_specs_cache[cache_key] = []
            return self._scan_area_tile_specs_cache[cache_key]
        if not all(k in self.scan_config for k in ('start_column', 'stop_column', 'start_row', 'stop_row')):
            self._scan_area_tile_specs_cache[cache_key] = []
            return self._scan_area_tile_specs_cache[cache_key]

        row_splits = int(split_config.get('rows', 1))
        col_splits = int(split_config.get('cols', 1))
        if row_splits < 1 or col_splits < 1:
            self.log.warning('Ignoring invalid map split configuration: rows and cols must be >= 1.')
            self._scan_area_tile_specs_cache[cache_key] = []
            return self._scan_area_tile_specs_cache[cache_key]
        if row_splits == 1 and col_splits == 1:
            self._scan_area_tile_specs_cache[cache_key] = []
            return self._scan_area_tile_specs_cache[cache_key]

        start_column = int(self.scan_config['start_column'])
        stop_column = int(self.scan_config['stop_column'])
        start_row = int(self.scan_config['start_row'])
        stop_row = int(self.scan_config['stop_row'])
        scan_cols = stop_column - start_column
        scan_rows = stop_row - start_row

        valid_cols = (scan_cols % col_splits == 0) and (scan_cols // col_splits > 2)
        valid_rows = (scan_rows % row_splits == 0) and (scan_rows // row_splits > 2)
        if not (valid_cols and valid_rows):
            self.log.warning(
                'Skipping zoomed map tiles for %s: scan area %d x %d is not compatible with splits rows=%d cols=%d. '
                'Each split must divide the scan area exactly and leave more than 2 pixels per tile.',
                self._active_plot_id, scan_rows, scan_cols, row_splits, col_splits
            )
            self._scan_area_tile_specs_cache[cache_key] = []
            return self._scan_area_tile_specs_cache[cache_key]

        col_step = scan_cols // col_splits
        row_step = scan_rows // row_splits
        tiles = []
        for row_index in range(row_splits):
            row_start = start_row + row_index * row_step
            row_stop = row_start + row_step
            for col_index in range(col_splits):
                col_start = start_column + col_index * col_step
                col_stop = col_start + col_step
                tiles.append({
                    'row_index': row_index,
                    'col_index': col_index,
                    'row_start': row_start,
                    'row_stop': row_stop,
                    'col_start': col_start,
                    'col_stop': col_stop,
                    'row_splits': row_splits,
                    'col_splits': col_splits,
                })
        self._scan_area_tile_specs_cache[cache_key] = tiles
        return self._scan_area_tile_specs_cache[cache_key]

    def _set_pixel_axis_ticks(self, ax):
        def get_ticks(low, high):
            start = int(round(min(low, high) - 0.5))
            stop = int(round(max(low, high) - 0.5))
            span = stop - start
            if span >= 512:
                step = 64
            elif span >= 256:
                step = 64
            elif span >= 128:
                step = 32
            elif span >= 64:
                step = 16
            elif span >= 32:
                step = 8
            elif span >= 16:
                step = 4
            elif span >= 8:
                step = 2
            else:
                step = 1

            labels = list(range(start, stop + 1, step))
            if labels[0] != start:
                labels.insert(0, start)
            if labels[-1] != stop:
                labels.append(stop)
            positions = [label + 0.5 for label in labels]
            return positions, labels

        x_positions, x_labels = get_ticks(self.plot_box_bounds[0], self.plot_box_bounds[1])
        y_positions, y_labels = get_ticks(self.plot_box_bounds[2], self.plot_box_bounds[3])
        ax.xaxis.set_major_locator(FixedLocator(x_positions))
        ax.yaxis.set_major_locator(FixedLocator(y_positions))
        ax.set_xticklabels([str(label) for label in x_labels])
        ax.set_yticklabels([str(label) for label in y_labels])

    def _integer_colorbar_ticks(self, z_min, z_max):
        start = int(math.floor(z_min))
        stop = int(math.ceil(z_max))
        span = max(1, stop - start)
        if span <= 16:
            step = 1
        elif span <= 32:
            step = 2
        elif span <= 64:
            step = 4
        elif span <= 128:
            step = 8
        else:
            step = int(math.ceil(span / 16.0))

        ticks = list(range(start, stop + 1, step))
        if ticks[0] != start:
            ticks.insert(0, start)
        if ticks[-1] != stop:
            ticks.append(stop)
        return np.array(ticks)

    def _resolve_map_z_limits(self, hist, z_min=None, z_max=None):
        if z_max == 'median':
            z_max = 2 * np.ma.median(hist)
        elif z_max == 'maximum':
            z_max = np.ma.max(hist)
        elif z_max is None:
            try:
                z_max = np.nanpercentile(hist.filled(np.nan), q=90)
                if np.any(hist[np.isfinite(hist)] > z_max):
                    z_max = 1.1 * z_max
            except TypeError:
                z_max = np.ma.max(hist)
        if z_max < 1 or hist.all() is np.ma.masked:
            z_max = 1.0

        if z_min is None:
            z_min = np.ma.min(hist)
            if z_min < 0:
                z_min = 0
        if z_min == z_max or hist.all() is np.ma.masked:
            z_min = 0
        return z_min, z_max

    def set_axis_ranges(self, axis_ranges):
        """Update per-plot axis overrides."""
        self.axis_ranges = axis_ranges or {}

    def get_available_plot_specs(self):
        """Return the named plot steps available for the current file."""
        specs = []
        scan_id = self.run_config['scan_id']
        if scan_id in ['dac_linearity_scan', 'adc_tuning']:
            specs.extend([
                ('parameter_page', 'Parameter page', self.create_parameter_page),
                ('dac_linearity', 'DAC linearity', self.create_dac_linearity_plot),
            ])
            return specs

        specs.append(('parameter_page', 'Parameter page', self.create_parameter_page))
        if self._monitoring_enabled_in_pdf():
            specs.extend([
                ('monitoring_summary', 'Monitoring summary', self.create_monitoring_summary_table),
                ('monitoring_main', 'Monitoring main page', self.create_monitoring_main_page),
            ])
        specs.append(('occupancy_map', 'Occupancy map', self.create_occupancy_map))
        if scan_id in SOURCE_LIKE_SCANS:
            specs.append(('fancy_occupancy', 'Fancy occupancy', self.create_fancy_occupancy))
        if scan_id in TOT_LIKE_SCANS:
            specs.extend([
                ('hit_pix', 'Hits per pixel', self.create_hit_pix_plot),
                ('tdac_plot', 'TDAC distribution', self.create_tdac_plot),
                ('tdac_map', 'TDAC map', self.create_tdac_map),
                ('tot_plot', 'ToT distribution', self.create_tot_plot),
                ('tot_map', 'ToT map', self.create_tot_map),
            ])
        if scan_id in THRESHOLD_LIKE_SCANS:
            specs.extend([
                ('tot_hist', 'ToT histogram', self.create_tot_hist),
                ('scurves', 'S-curves', self.create_scurves_plot),
                ('threshold_plot', 'Threshold distribution', self.create_threshold_plot),
                ('stacked_threshold', 'Stacked threshold', self.create_stacked_threshold_plot),
                ('threshold_map', 'Threshold map', self.create_threshold_map),
                ('noise_plot', 'Noise distribution', self.create_noise_plot),
                ('noise_map', 'Noise map', self.create_noise_map),
            ])
        if scan_id == 'global_threshold_tuning':
            specs.extend([
                ('scurves', 'S-curves', self.create_scurves_plot),
                ('threshold_plot', 'Threshold distribution', self.create_threshold_plot),
                ('threshold_map', 'Threshold map', self.create_threshold_map),
                ('noise_plot', 'Noise distribution', self.create_noise_plot),
                ('noise_map', 'Noise map', self.create_noise_map),
            ])
        if self.clustered:
            specs.extend([
                ('cluster_tot', 'Cluster ToT', self.create_cluster_tot_plot),
                ('cluster_shape', 'Cluster shape', self.create_cluster_shape_plot),
                ('cluster_size', 'Cluster size', self.create_cluster_size_plot),
            ])
        if self.HistTdcStatus is not None:
            specs.append(('tdc_status', 'TDC status', self.create_tdc_status_plot))
        return specs

    def list_available_plots(self):
        """Return the available plot ids in execution order."""
        return [plot_id for plot_id, _, _ in self.get_available_plot_specs()]

    def list_available_plot_details(self):
        """Return available plot ids together with user-facing labels."""
        return [(plot_id, label) for plot_id, label, _ in self.get_available_plot_specs()]

    def create_selected_plots(self, selected_plots=None):
        """Run all available plots or only the selected named plots."""
        if self.skip_plotting:
            return
        specs = self.get_available_plot_specs()
        if selected_plots is not None:
            requested = set(selected_plots)
            unknown = sorted(requested - {plot_id for plot_id, _, _ in specs})
            if unknown:
                raise ValueError(f'Unknown plot ids: {", ".join(unknown)}')
            specs = [spec for spec in specs if spec[0] in requested]
        self._run_named_plot_steps(specs, description='Plotting')

    ''' User callable plotting functions '''
    def create_standard_plots(self):
        if self.skip_plotting:
            return
        self.log.info('Creating selected plots...')
        self.create_selected_plots()

    def create_parameter_page(self):
        try:
            self._plot_parameter_page()
        except Exception:
            self.log.error('Could not create parameter page!')

    def create_tuning_plots(self, include_tdac=True):
        """Create a smaller tuning-oriented plot set."""
        if self.skip_plotting:
            return
        self.log.info('Creating tuning plots...')
        steps = [('parameter_page', 'Parameter page', self.create_parameter_page)]
        if self._monitoring_enabled_in_pdf():
            steps.extend([
                ('monitoring_summary', 'Monitoring summary', self.create_monitoring_summary_table),
                ('monitoring_main', 'Monitoring main page', self.create_monitoring_main_page),
            ])
        steps.extend([
            ('occupancy_map', 'Occupancy map', self.create_occupancy_map),
            ('hit_pix', 'Hits per pixel', self.create_hit_pix_plot),
        ])
        if include_tdac:
            steps.extend([
                ('tdac_plot', 'TDAC distribution', self.create_tdac_plot),
                ('tdac_map', 'TDAC map', self.create_tdac_map),
            ])
        steps.extend([
            ('tot_plot', 'ToT distribution', self.create_tot_plot),
            ('tot_map', 'ToT map', self.create_tot_map),
        ])
        self._run_named_plot_steps(steps, description='Tuning plots')

    def _monitoring_enabled_in_pdf(self):
        if not self.monitoring_group:
            return False
        if not self.monitoring_cfg:
            return True
        return self.monitoring_cfg.get('include_in_pdf', True)

    def _run_named_plot_steps(self, steps, description='Plotting'):
        """Run a list of named plot callables with optional progress output."""
        if not steps:
            return
        if not self.show_progress:
            for plot_id, _, func in steps:
                self._active_plot_id = plot_id
                try:
                    func()
                finally:
                    self._active_plot_id = None
            return
        with tqdm(total=len(steps), desc=description, unit='plot') as pbar:
            for plot_id, label, func in steps:
                pbar.set_postfix_str(label)
                self._active_plot_id = plot_id
                try:
                    func()
                finally:
                    self._active_plot_id = None
                pbar.update(1)

    def _get_axis_override(self, suffix):
        """Return the axis override dict for a given plot suffix."""
        keys = []
        if self._active_plot_id is not None:
            keys.append(self._active_plot_id)
        if suffix is not None:
            keys.append(suffix)

        override = {}
        for key in keys:
            if key in self.axis_ranges:
                override.update(self.axis_ranges[key])
        return override

    def _apply_axis_override(self, ax, suffix, allow_x=True, allow_y=True):
        """Apply x/y range overrides after a plot is created."""
        override = self._get_axis_override(suffix)
        if allow_x and 'x' in override:
            ax.set_xlim(*override['x'])
        if allow_y and 'y' in override:
            ax.set_ylim(*override['y'])

    def _apply_map_range_override(self, suffix, z_min, z_max):
        """Apply z-range overrides before drawing a map."""
        override = self._get_axis_override(suffix)
        if 'z' in override:
            z_min, z_max = override['z']
        return z_min, z_max

    def _sanitize_distribution_data(self, data):
        data = np.ma.masked_invalid(np.ma.array(data, copy=False))
        return data.compressed()

    def _filled_projection(self, values):
        values = np.ma.masked_invalid(np.ma.array(values, copy=False))
        return values.filled(0)

    def _decode_bytes(self, value):
        if isinstance(value, bytes):
            return value.decode('utf-8')
        return value

    def _read_monitoring_table(self, table):
        if isinstance(table, dict) and 'data' in table:
            data = table['data']
            label_map = table.get('label_map', {})
            unit_map = table.get('unit_map', {})
        else:
            try:
                label_map = json.loads(table.attrs.label_map)
            except Exception:
                label_map = {}
            try:
                unit_map = json.loads(table.attrs.unit_map)
            except Exception:
                unit_map = {}
            data = table[:]

        ts = data['timestamp']
        series = {}
        for field in data.dtype.names:
            if field == 'timestamp':
                continue
            label = label_map.get(field, field)
            series[label] = data[field]

        return ts, series, label_map, unit_map

    def _snapshot_monitoring_table(self, table):
        try:
            label_map = json.loads(table.attrs.label_map)
        except Exception:
            label_map = {}
        try:
            unit_map = json.loads(table.attrs.unit_map)
        except Exception:
            unit_map = {}
        return {
            'data': table[:],
            'label_map': label_map,
            'unit_map': unit_map
        }

    def create_monitoring_summary_table(self):
        if self.monitoring_summary is None:
            return
        try:
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.axis('off')

            rows = []
            for row in self.monitoring_summary:
                attr = self._decode_bytes(row['attribute'])
                unit = self._decode_bytes(row['unit'])
                mean = row['mean']
                # Keep only main monitoring values for the PDF
                keep = False
                if 'NTC' in attr:
                    keep = True
                if attr in ['HV_V', 'HV_I', 'PWELL_V', 'PWELL_I', 'PSUB_PWELL_V', 'PSUB_PWELL_I']:
                    keep = True
                if not keep:
                    continue
                rows.append([attr, f'{mean:.3g}', unit])

            if not rows:
                return

            labels = ['Attribute', 'Mean', 'Unit']
            widths = [0.65, 0.20, 0.15]
            table = ax.table(cellText=rows, colWidths=widths, colLabels=labels, cellLoc='left', loc='center')
            table.scale(1.0, 1.0)
            table.auto_set_font_size(False)
            for key, cell in table.get_celld().items():
                cell.set_fontsize(6)
                if key[0] == 0:
                    cell.set_color('#ffb300')
                    cell.set_fontsize(7)

            ax.set_title('Monitoring summary (scan window)', fontsize=10)
            start_str = self._format_monitoring_timestamp(self.monitoring_scan_start)
            stop_str = self._format_monitoring_timestamp(self.monitoring_scan_stop)
            if start_str and stop_str:
                ax.text(0.0, 0.92, f'Start: {start_str}    Stop: {stop_str}', transform=ax.transAxes, fontsize=7)
            self._save_plots(fig, suffix='monitoring_summary')
        except Exception:
            self.log.error('Could not create monitoring summary table!')

    def create_monitoring_time_series(self):
        if not self.monitoring_group:
            return
        try:
            if 'env' in self.monitoring_tables:
                self._plot_env_time_series()
            if 'power' in self.monitoring_tables:
                self._plot_power_time_series()
        except Exception:
            self.log.error('Could not create monitoring plots!')

    def create_monitoring_main_page(self):
        if not self.monitoring_group:
            return
        try:
            self._plot_ntc_time_series()
            self._plot_currents_time_series(
                wanted_labels=['HV_I'],
                title='HV current vs time',
                suffix='monitoring_current_hv'
            )
            self._plot_currents_time_series(
                wanted_labels=['PWELL_I', 'PSUB_PWELL_I'],
                title='PWELL and PSUB-PWELL currents vs time',
                suffix='monitoring_current_pwell_psub'
            )
        except Exception:
            self.log.error('Could not create monitoring main page!')

    def _format_monitoring_timestamp(self, timestamp):
        if timestamp is None:
            return None
        try:
            return datetime.datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return None

    def _plot_ntc_time_series(self):
        table = self.monitoring_tables.get('env')
        if not table:
            return
        ts, series, _, _ = self._read_monitoring_table(table)
        if ts.size == 0 or not series:
            return
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        t0 = ts[0]
        time_s = ts - t0
        plotted = False
        for label, values in series.items():
            if 'NTC' in label:
                ax.plot(time_s, values, label=label)
                plotted = True
        if not plotted:
            return
        ax.set_title('NTC temperature vs time')
        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel('Temperature [°C]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        self._save_plots(fig, suffix='monitoring_ntc')

    def _plot_currents_time_series(self, wanted_labels, title, suffix):
        table = self.monitoring_tables.get('power')
        if not table:
            return
        ts, series, _, _ = self._read_monitoring_table(table)
        if ts.size == 0 or not series:
            return
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        t0 = ts[0]
        time_s = ts - t0
        plotted = False
        for label, values in series.items():
            if label in wanted_labels:
                ax.plot(time_s, values, label=label)
                plotted = True
        if not plotted:
            return
        ax.set_title(title)
        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel('Current [A]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        self._save_plots(fig, suffix=suffix)

    def _plot_env_time_series(self):
        table = self.monitoring_tables.get('env')
        if not table:
            return
        ts, series, _, _ = self._read_monitoring_table(table)
        if ts.size == 0 or not series:
            return
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        t0 = ts[0]
        time_s = ts - t0
        for label, values in series.items():
            ax.plot(time_s, values, label=label)
        ax.set_title('Environmental monitor vs time')
        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel('Temperature')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        self._save_plots(fig, suffix='monitoring_env')

    def _plot_power_time_series(self):
        table = self.monitoring_tables.get('power')
        if not table:
            return
        ts, series, _, _ = self._read_monitoring_table(table)
        if ts.size == 0 or not series:
            return
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        t0 = ts[0]
        time_s = ts - t0

        wanted = ['HV_I', 'PWELL_I', 'PSUB_PWELL_I']
        plotted = False
        for label in series.keys():
            norm = label.replace(' ', '').replace('-', '').replace('_', '').upper()
            if label in wanted or label.replace(' ', '') in wanted:
                ax.plot(time_s, series[label], label=label)
                plotted = True
            elif norm in ['HVI', 'PWELLI', 'PSUBI']:
                ax.plot(time_s, series[label], label=label)
                plotted = True

        if not plotted:
            for label, values in series.items():
                if label.endswith('_I') or 'current' in label.lower():
                    ax.plot(time_s, values, label=label)
                    plotted = True

        if not plotted:
            return

        ax.set_title('Currents vs time')
        ax.set_xlabel('Time since start [s]')
        ax.set_ylabel('Current [A]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        self._save_plots(fig, suffix='monitoring_currents')

    def create_occupancy_map(self):
        try:
            if self.run_config['scan_id'] in ['threshold_scan', 'fast_threshold_scan', 'autorange_threshold_scan', 'global_threshold_tuning', 'injection_delay_scan', 'in_time_threshold_scan', 'injection_delay_scan', 'crosstalk_scan']:
                title = 'Integrated occupancy'
                z_max = 'maximum'
            else:
                title = 'Occupancy'
                z_max = None

            hist = np.ma.masked_array(self.HistOcc[:].sum(axis=2), self.enable_mask).T
            z_min, z_max = self._resolve_map_z_limits(hist, z_max=z_max)
            self._plot_occupancy(hist=hist,
                                 z_min=z_min,
                                 z_max=z_max,
                                 suffix='occupancy',
                                 title=title,
                                 colorbar_scientific=True)
            self._plot_scan_area_map(hist=hist,
                                     z_min=z_min,
                                     z_max=z_max,
                                     suffix='occupancy_scan_area',
                                     title=title + ' scan area',
                                     colorbar_scientific=True)
            self._plot_scan_area_map_tiles(hist=hist,
                                           z_min=z_min,
                                           z_max=z_max,
                                           suffix='occupancy_scan_area_zoom',
                                           title=title + ' scan area',
                                           colorbar_scientific=True)
        except Exception:
            self.log.error('Could not create occupancy map!')

    def create_fancy_occupancy(self):
        try:
            self._plot_fancy_occupancy(hist=np.ma.masked_array(self.HistOcc[:].sum(axis=2), self.enable_mask).T)
        except Exception:
            self.log.error('Could not create fancy occupancy plot!')

    def create_tot_plot(self):
        ''' Create 1D tot plot '''
        try:
            hist_tot = self._sum_hist_tot()
            title = ('Time-over-Threshold distribution ($\\Sigma$ = {0:1.0f})'.format(np.sum(hist_tot)))
            self._plot_1d_hist(hist=hist_tot,
                               title=title,
                               log_y=False,
                               plot_range=range(0, self.HistTot.shape[3]),
                            #    plot_range=range(0, 40),
                               x_axis_title='ToT code',
                               y_axis_title='# of hits',
                               color='b',
                               suffix='tot')
        except Exception:
            self.log.error('Could not create tot plot!')

    def create_tot_map(self):
        '''Create average ToT map from HistTot using chunked reads.'''
        try:
            total, weighted = self._sum_hist_tot_per_pixel()
            if not np.any(total):
                return
            mean_tot = np.zeros(total.shape, dtype=float)
            np.divide(weighted, total, out=mean_tot, where=total > 0)
            z_min = 0
            z_max = int(np.ceil(np.nanmax(mean_tot[total > 0])))
            hist = np.ma.masked_array(mean_tot, self.enable_mask).T
            self._plot_occupancy(hist=hist,
                                 title='Average ToT map',
                                 z_label='ToT code',
                                 z_min=z_min,
                                 z_max=z_max,
                                 suffix='tot_map',
                                 colorbar_integer=True,
                                 extend_upper_bound=False)
            self._plot_scan_area_map(hist=hist,
                                     title='Average ToT map scan area',
                                     z_label='ToT code',
                                     z_min=z_min,
                                     z_max=z_max,
                                     suffix='tot_map_scan_area',
                                     colorbar_integer=True,
                                     extend_upper_bound=False)
            self._plot_scan_area_map_tiles(hist=hist,
                                           title='Average ToT map scan area',
                                           z_label='ToT code',
                                           z_min=z_min,
                                           z_max=z_max,
                                           suffix='tot_map_scan_area_zoom',
                                           colorbar_integer=True,
                                           extend_upper_bound=False)
        except Exception:
            self.log.error('Could not create average ToT map!')

    def create_tot_hist(self):
        try:
            data = self._sum_hist_tot_by_scan_param()
            self._plot_2d_param_hist(hist=data,
                                     y_max=self.HistTot.shape[3],
                                     scan_parameters=self.scan_parameter_range,
                                     electron_axis=False,
                                     scan_parameter_name='$\\Delta$ VCAL',
                                     title='ToT Scan Parameter Histogram',
                                     ylabel='ToT code',
                                     suffix='tot_param_hist')
        except Exception as e:
            self.log.error('Could not create tot param histogram plot! ({0})'.format(e))

    def create_scurves_plot(self, scan_parameter_name='Scan parameter'):
        try:
            if self.run_config['scan_id'] == 'injection_delay_scan':
                scan_parameter_name = 'Finedelay [LSB]'
                plot_electron_axis = False
                scan_parameter_range = range(0, 16)
            elif self.run_config['scan_id'] == 'global_threshold_tuning':
                scan_parameter_name = self.scan_config['VTH_name']
                plot_electron_axis = False
                scan_parameter_range = self.scan_parameter_range
            else:
                scan_parameter_name = '$\\Delta$ VCAL'
                scan_parameter_range = self.scan_parameter_range
                plot_electron_axis = self.plot_electron_axis

            params = [{'scurves': self.HistOcc[:].ravel().reshape((self.rows * self.cols, -1)).T,
                       'scan_parameters': scan_parameter_range,
                       'electron_axis': plot_electron_axis,
                       'scan_parameter_name': scan_parameter_name}]

            for param in params:
                self._plot_scurves(**param)
        except Exception as e:
            self.log.error('Could not create scurve plot! ({0})'.format(e))

    def create_threshold_plot(self, logscale=False, scan_parameter_name='Scan parameter'):
        try:
            title = 'Threshold distribution for enabled pixels'
            threshold_data = self.ThresholdMap[self.Chi2Sel].T
            if self.run_config['scan_id'] == 'injection_delay_scan':
                scan_parameter_name = 'Finedelay [LSB]'
                plot_electron_axis = False
                plot_range = range(0, 16)
                title = 'Fine delay distribution for enabled pixels'
            elif self.run_config['scan_id'] == 'global_threshold_tuning':
                plot_range = self._threshold_distribution_range(threshold_data, margin=5)
                scan_parameter_name = self.scan_config['VTH_name']
                plot_electron_axis = False
            else:
                plot_range = self._threshold_distribution_range(threshold_data, margin=5)
                scan_parameter_name = '$\\Delta$ VCAL'
                plot_electron_axis = self.plot_electron_axis

            self._plot_distribution(threshold_data,
                                    plot_range=plot_range,
                                    electron_axis=plot_electron_axis,
                                    x_axis_title=scan_parameter_name,
                                    title=title,
                                    log_y=logscale,
                                    y_axis_title='# of pixels',
                                    print_failed_fits=True,
                                    suffix='threshold_distribution')
        except Exception as e:
            self.log.error('Could not create threshold plot! ({0})'.format(e))

    def _threshold_distribution_range(self, data, margin=5):
        data = np.ma.masked_invalid(np.ma.array(data, copy=False))
        data = data.compressed()
        data = data[data < 1e5]
        if data.size == 0:
            return self.scan_parameter_range

        scan_min = min(self.scan_parameter_range)
        scan_max = max(self.scan_parameter_range)
        lower = max(scan_min, math.floor(float(np.nanmin(data)) - margin))
        upper = min(scan_max, math.ceil(float(np.nanmax(data)) + margin))
        if upper <= lower:
            upper = lower + 1
        step = 1 if upper - lower <= 100 else max(1, int(math.ceil((upper - lower) / 100.0)))
        return np.arange(lower, upper + step, step)

    def create_stacked_threshold_plot(self, scan_parameter_name='Scan parameter'):
        try:
            min_tdac, max_tdac, range_tdac, _ = (1, 7, 7, 1)
            threshold_data = self.ThresholdMap[self.Chi2Sel].T

            plot_range = self._threshold_distribution_range(threshold_data, margin=5)
            if self.run_config['scan_id'] == 'global_threshold_tuning':
                scan_parameter_name = self.scan_config['VTH_name']
                plot_electron_axis = False
            else:
                scan_parameter_name = '$\\Delta$ VCAL'
                plot_electron_axis = self.plot_electron_axis

            self._plot_stacked_threshold(data=threshold_data,
                                         tdac_mask=self.tdac_mask[self.Chi2Sel].T,
                                         plot_range=plot_range,
                                         electron_axis=plot_electron_axis,
                                         x_axis_title=scan_parameter_name,
                                         y_axis_title='# of pixels',
                                         title='Threshold distribution for enabled pixels',
                                         suffix='tdac_threshold_distribution',
                                         min_tdac=min(min_tdac, max_tdac),
                                         max_tdac=max(min_tdac, max_tdac),
                                         range_tdac=range_tdac, centered_ticks=True)
        except Exception:
            self.log.error('Could not create stacked threshold plot!')

    def create_threshold_map(self):
        try:
            mask = self.enable_mask.copy()
            sel = self.Chi2Map[:] > 0.  # Mask not converged fits (chi2 = 0)
            mask[~sel] = True
            if self.run_config['scan_id'] == 'injection_delay_scan':
                plot_electron_axis = False
                use_electron_offset = False
                z_label = 'Finedelay [LSB]'
                title = 'Injection Delay'
                z_min = 0
                z_max = 16
            else:
                plot_electron_axis = self.plot_electron_axis
                use_electron_offset = False
                z_label = 'Threshold'
                title = 'Threshold'
                z_min = None
                z_max = None

            hist = np.ma.masked_array(self.ThresholdMap, mask).T
            z_min, z_max = self._resolve_map_z_limits(hist, z_min=z_min, z_max=z_max)
            self._plot_occupancy(hist=hist,
                                 electron_axis=plot_electron_axis,
                                 z_label=z_label,
                                 title=title,
                                 use_electron_offset=use_electron_offset,
                                 show_sum=False,
                                 z_min=z_min,
                                 z_max=z_max,
                                 suffix='threshold_map',
                                 colorbar_tick_count=6)
            self._plot_scan_area_map(hist=hist,
                                     electron_axis=plot_electron_axis,
                                     z_label=z_label,
                                     title=title + ' scan area',
                                     use_electron_offset=use_electron_offset,
                                     show_sum=False,
                                     z_min=z_min,
                                     z_max=z_max,
                                     suffix='threshold_map_scan_area',
                                     colorbar_tick_count=6)
            self._plot_scan_area_map_tiles(hist=hist,
                                           electron_axis=plot_electron_axis,
                                           z_label=z_label,
                                           title=title + ' scan area',
                                           use_electron_offset=use_electron_offset,
                                           show_sum=False,
                                           z_min=z_min,
                                           z_max=z_max,
                                           suffix='threshold_map_scan_area_zoom',
                                           colorbar_tick_count=6)
        except Exception:
            self.log.error('Could not create threshold map!')

    def create_noise_plot(self, logscale=False, scan_parameter_name='Scan parameter'):
        try:
            mask = self.enable_mask.copy()
            sel = self.Chi2Map[:] > 0.  # Mask not converged fits (chi2 = 0)
            mask[~sel] = True

            plot_range = None
            if self.run_config['scan_id'] in ['threshold_scan', 'fast_threshold_scan', 'in_time_threshold_scan', 'autorange_threshold_scan', 'crosstalk_scan']:
                scan_parameter_name = '$\\Delta$ VCAL'
                plot_electron_axis = self.plot_electron_axis
            elif self.run_config['scan_id'] == 'global_threshold_tuning':
                scan_parameter_name = self.scan_config['VTH_name']
                plot_electron_axis = False
            elif self.run_config['scan_id'] == 'injection_delay_scan':
                scan_parameter_name = 'Finedelay [LSB]'
                plot_electron_axis = False

            self._plot_distribution(np.ma.masked_array(self.NoiseMap, mask).T,
                                    title='Noise distribution for enabled pixels',
                                    plot_range=plot_range,
                                    electron_axis=plot_electron_axis,
                                    use_electron_offset=False,
                                    x_axis_title=scan_parameter_name,
                                    y_axis_title='# of pixels',
                                    log_y=logscale,
                                    print_failed_fits=True,
                                    suffix='noise_distribution')
        except Exception:
            self.log.error('Could not create noise plot!')

    def create_noise_map(self):
        try:
            mask = self.enable_mask.copy()
            sel = self.Chi2Map[:] > 0.  # Mask not converged fits (chi2 = 0)
            mask[~sel] = True
            z_label = 'Noise'
            title = 'Noise'
            plot_electron_axis = self.plot_electron_axis

            if self.run_config['scan_id'] == 'injection_delay_scan':
                z_label = 'Finedelay [LSB]'
                title = 'Injection Delay Noise'
                plot_electron_axis = False
            hist = np.ma.masked_array(self.NoiseMap, mask).T
            z_min, z_max = self._resolve_map_z_limits(hist, z_max='median')
            self._plot_occupancy(hist=hist,
                                 electron_axis=plot_electron_axis,
                                 use_electron_offset=False,
                                 z_label=z_label,
                                 z_min=z_min,
                                 z_max=z_max,
                                 title=title,
                                 show_sum=False,
                                 suffix='noise_map',
                                 colorbar_tick_count=6)
            self._plot_scan_area_map(hist=hist,
                                     electron_axis=plot_electron_axis,
                                     use_electron_offset=False,
                                     z_label=z_label,
                                     z_min=z_min,
                                     z_max=z_max,
                                     title=title + ' scan area',
                                     show_sum=False,
                                     suffix='noise_map_scan_area',
                                     colorbar_tick_count=6)
            self._plot_scan_area_map_tiles(hist=hist,
                                           electron_axis=plot_electron_axis,
                                           use_electron_offset=False,
                                           z_label=z_label,
                                           z_min=z_min,
                                           z_max=z_max,
                                           title=title + ' scan area',
                                           show_sum=False,
                                           suffix='noise_map_scan_area_zoom',
                                           colorbar_tick_count=6)
        except Exception:
            self.log.error('Could not create noise map!')

    def create_tdac_plot(self):
        try:
            mask = self.enable_mask.copy()
            min_tdac, max_tdac, _, tdac_incr = (0, 8, 8, 1)
            plot_range = range(min_tdac, max_tdac + tdac_incr, tdac_incr)
            self._plot_distribution(self.tdac_mask[~mask].T,
                                    plot_range=plot_range,
                                    title='TDAC distribution for enabled pixels',
                                    x_axis_title='TDAC',
                                    y_axis_title='# of pixels',
                                    align='center',
                                    suffix='tdac_distribution')
        except Exception:
            self.log.error('Could not create TDAC plot!')

    def create_tdac_map(self):
        try:
            mask = self.enable_mask.copy()
            min_tdac, max_tdac = (1, 7)
            hist = np.ma.masked_array(self.tdac_mask, mask).T
            self._plot_fancy_occupancy(hist=hist,
                                       title='TDAC map',
                                       z_label='TDAC',
                                       z_min=min(min_tdac, max_tdac),
                                       z_max=max(min_tdac, max_tdac),
                                       log_z=False, centered_ticks=True,
                                       norm_projection=True)
            self._plot_scan_area_fancy_map(hist=hist,
                                           title='TDAC map scan area',
                                           z_label='TDAC',
                                           z_min=min(min_tdac, max_tdac),
                                           z_max=max(min_tdac, max_tdac),
                                           log_z=False,
                                           centered_ticks=True,
                                           norm_projection=True,
                                           suffix='tdac_map_scan_area')
            self._plot_scan_area_map_tiles(hist=hist,
                                           title='TDAC map scan area',
                                           z_label='TDAC',
                                           z_min=min(min_tdac, max_tdac),
                                           z_max=max(min_tdac, max_tdac),
                                           log_z=False,
                                           centered_ticks=True,
                                           norm_projection=True,
                                           suffix='tdac_map_scan_area_zoom',
                                           plot_kind='fancy')
        except Exception:
            self.log.error('Could not create TDAC map!')

    def create_chi2_map(self):
        try:
            mask = self.enable_mask.copy()
            chi2 = self.Chi2Map[:]
            sel = chi2 > 0.  # Mask not converged fits (chi2 = 0)
            mask[~sel] = True

            hist = np.ma.masked_array(chi2, mask).T
            z_min, z_max = self._resolve_map_z_limits(hist, z_max='median')
            self._plot_occupancy(hist=hist,
                                 z_label='Chi2/ndf.',
                                 z_min=z_min,
                                 z_max=z_max,
                                 title='Chi2 over ndf of S-Curve fits',
                                 show_sum=False,
                                 suffix='chi2_map',
                                 colorbar_tick_count=6)
            self._plot_scan_area_map(hist=hist,
                                     z_label='Chi2/ndf.',
                                     z_min=z_min,
                                     z_max=z_max,
                                     title='Chi2 over ndf of S-Curve fits scan area',
                                     show_sum=False,
                                     suffix='chi2_map_scan_area',
                                     colorbar_tick_count=6)
        except Exception:
            self.log.error('Could not create chi2 map!')

    def create_cluster_size_plot(self):
        ''' Create 1D cluster size plot '''
        try:
            self._plot_1d_hist(hist=self.HistClusterSize[:], title='Cluster size',
                               log_y=False, plot_range=range(0, 10),
                               x_axis_title='Cluster size',
                               y_axis_title='# of clusters', suffix='cluster_size')
        except Exception:
            self.log.error('Could not create cluster size plot!')

    def create_cluster_tot_plot(self):
        ''' Create 1D cluster ToT plot '''
        try:
            if np.max(np.nonzero(self.HistClusterTot)) < 128:
                plot_range = range(0, 128)
                #plot_range = range(0, 50)
                print('step1')
            else:
                plot_range = range(0, np.max(np.nonzero(self.HistClusterTot)))
                #plot_range = range(0, 50)
                print('step2')

            #tot_calib_file = self.configuration['scan'].get('tot_calib_file', None)

            tot_calib_file = None
            #tot_calib_file ='/home/labb2/tj-monopix2-daq-development/tjmonopix2/scans/output_data/module_0/chip_0/20241212_162859_calibrate_tot_interpreted.h5'
            print('tot_calib file ',tot_calib_file)
            print('plot range ',plot_range)
            #plot_range = [x * self.electron_conversion for x in plot_range]
            #print('new range ',plot_range)
            if tot_calib_file is not None:
                print('step3')
                x_axis_title = 'Cluster charge [DAC]'
                # x_axis_title = 'Cluster charge [e⁻]'
                # plot_range = [x * self.electron_conversion for x in plot_range]
                #plot_range = plot_range * self.electron_conversion
            else:
                x_axis_title = 'Cluster ToT [25 ns]'
                print('step4')

            self._plot_1d_hist(hist=self.HistClusterTot[:], title='Cluster ToT',
                               log_y=True, plot_range=plot_range,
                               x_axis_title=x_axis_title,
                               y_axis_title='# of clusters', suffix='cluster_tot')
            print('step_last')
        except Exception:
            self.log.error('Could not create cluster TOT plot!')

    def create_cluster_shape_plot(self):
        try:
            self._plot_cl_shape(self.HistClusterShape[:])
        except Exception:
            self.log.error('Could not create cluster shape plot!')

    def create_hit_pix_plot(self):
        try:
            occ_1d = np.ma.masked_array(self.HistOcc[:].sum(axis=2), self.enable_mask).ravel()

            if occ_1d.sum() == 0:
                plot_range = np.arange(0, 100, 1)
            else:
                plot_range = np.arange(0, occ_1d.max() + 0.05 * occ_1d.max(), occ_1d.max() / 100)
            self._plot_distribution(data=occ_1d,
                                    plot_range=plot_range,
                                    title='Hits per Pixel',
                                    x_axis_title='# of Hits',
                                    y_axis_title='# of Pixel',
                                    log_y=True,
                                    align='center',
                                    fit_gauss=False,
                                    suffix='hit_pix')
        except Exception:
            self.log.error('Could not create hits per pixel plot!')

    '''Internal functions not meant to be called by user'''

    def _mask_disabled_pixels(self, enable_mask, scan_config):
        mask = np.invert(enable_mask)
        mask[:scan_config['start_column'], :] = True
        mask[scan_config['stop_column']:, :] = True
        mask[:, :scan_config['start_row']] = True
        mask[:, scan_config['stop_row']:] = True

        return mask

    def _save_plots(self, fig, suffix=None, tight=False):
        """Save the current figure to the configured outputs."""
        if self.out_file is None:
            return
        increase_count = False
        bbox_inches = 'tight' if tight else ''
        if suffix is None:
            suffix = str(self.plot_cnt)
        self.out_file.savefig(fig, bbox_inches=bbox_inches)
        if self.save_png:
            fig.savefig(self.filename[:-4] + '_' + suffix + '.png', bbox_inches=bbox_inches)
            increase_count = True
        if self.save_single_pdf:
            fig.savefig(self.filename[:-4] + '_' + suffix + '.pdf', bbox_inches=bbox_inches)
            increase_count = True
        if increase_count:
            self.plot_cnt += 1

    def _add_text(self, fig):
        fig.subplots_adjust(top=0.85)
        y_coord = 0.92
        fig.text(0.1, y_coord, '{0} {1}'.format("TJ-Monopix2", self.level), fontsize=12, color=OVERTEXT_COLOR, transform=fig.transFigure)
        if self._module_type is None:
            module_text = 'Chip S/N: {0}'.format(self.run_config['chip_sn'])
        else:
            module_text = 'Module: {0}'.format(self.module_settings['identifier'])
        fig.text(0.7, y_coord, module_text, fontsize=12, color=OVERTEXT_COLOR, transform=fig.transFigure)
        if self.internal:
            fig.text(0.1, 1, 'Internal', fontsize=16, color='r', rotation=45, bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.7), transform=fig.transFigure)

    def _convert_to_e(self, dac, use_offset=False):
        if use_offset:
            e = dac * self.calibration['e_conversion_slope'] + self.calibration['e_conversion_offset']
            de = math.sqrt((dac * self.calibration['e_conversion_slope_error'])**2 + self.calibration['e_conversion_offset_error']**2)
        else:
            e = dac * self.electron_conversion
            de = dac * self.electron_conversion * 0.15  # TODO: Assume generic 15% error of conversion factor, need precise measurement.
        return e, de

    def _add_electron_axis(self, fig, ax, use_electron_offset=False):
        fig.subplots_adjust(top=0.75)
        ax.title.set_position([.5, 1.15])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        fig.canvas.draw()
        ax2 = ax.twiny()

        xticks = []
        for t in ax.get_xticks(minor=False):
            xticks.append(int(self._convert_to_e(float(t), use_offset=use_electron_offset)[0]))

        l1 = ax.get_xlim()
        l2 = ax2.get_xlim()

        def f(x):
            return l2[0] + (x - l1[0]) / (l1[1] - l1[0]) * (l2[1] - l2[0])

        ticks = f(ax.get_xticks())
        ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax2.set_xticks(ticks)
        ax2.set_xlabel('Electrons', labelpad=7)
        ax2.set_xticklabels(xticks)
        return ax2

    def _plot_parameter_page(self):
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.axis('off')

        scan_id = self.run_config['scan_id']
        run_name = self.run_config['run_name']
        if self._module_type is None:
            file_type = 'chip'
            chip_sn = self.run_config['chip_sn']
        else:
            file_type = 'module'
            chip_sn = self.run_config['module']

        sw_ver = self.run_config['software_version']
        timestamp = datetime.datetime.strptime(' '.join(run_name.split('_')[:2]), '%Y%m%d %H%M%S')  # FIXME: Workaround while there is no timestamp saved in h5 file

        text = 'This is a TJ-Monopix2 {0} for {1} {2}.\nRun {3} was started {4}.'.format(scan_id, file_type, chip_sn, run_name, timestamp)

        ax.text(0.01, 0.9, text, fontsize=10)
        ax.text(-0.1, -0.11, 'Software version: {0}'.format(sw_ver), fontsize=3)

        if 'maskfile' in self.scan_config.keys() and self.scan_config['maskfile'] is not None and not self.scan_config['maskfile'] == 'None':
            ax.text(0.01, -0.05, 'Maskfile:\n{0}'.format(self.scan_config['maskfile']), fontsize=6)

        scan_config_dict = OrderedDict()
        dac_dict = OrderedDict()
        scan_config_trg_tdc_dict = OrderedDict()

        exclude_run_conf_items = ['scan_id', 'run_name', 'timestamp', 'chip_sn', 'software_version', 'maskfile', 'TDAC']
        run_conf_trg_tdc = ['TRIGGER_MODE', 'TRIGGER_SELECT', 'TRIGGER_INVERT', 'TRIGGER_LOW_TIMEOUT', 'TRIGGER_VETO_SELECT',
                            'TRIGGER_HANDSHAKE_ACCEPT_WAIT_CYCLES', 'DATA_FORMAT', 'EN_TLU_VETO', 'TRIGGER_DATA_DELAY',
                            'EN_WRITE_TIMESTAMP', 'EN_TRIGGER_DIST', 'EN_NO_WRITE_TRIG_ERR', 'EN_INVERT_TDC', 'EN_INVERT_TRIGGER']

        for key, value in sorted(self.scan_config.items()):
            if key not in (exclude_run_conf_items + run_conf_trg_tdc):
                if key == 'module' and value.startswith('module_'):  # Nice formatting
                    value = value.split('module_')[1]
                if key == 'trigger_pattern':  # Nice formatting
                    value = hex(value)
                scan_config_dict[key] = value
            if key in run_conf_trg_tdc:
                scan_config_trg_tdc_dict[key] = value

        for flavor in DACS.keys():
            dac_dict[flavor] = OrderedDict()
            for reg, value in self.registers.items():
                if any(reg.startswith(dac) for dac in DACS[flavor]):
                    dac_dict[flavor][reg] = value
        tb_list = []
        for i in range(max(len(scan_config_dict), len(dac_dict['TJMONOPIX2']), len(scan_config_dict))):
            try:
                key1 = list(scan_config_dict.keys())[i]
                value1 = scan_config_dict[key1]
            except IndexError:
                key1 = ''
                value1 = ''
            try:
                key2 = list(dac_dict['TJMONOPIX2'].keys())[i]
                value2 = dac_dict['TJMONOPIX2'][key2]
            except IndexError:
                key2 = ''
                value2 = ''

            tb_list.append([key1, value1, '', key2, value2, ''])

        widths = [0.18, 0.10, 0.03, 0.18, 0.10, 0.03]
        labels = ['Scan config', 'Value', '', 'TJ-Monopix2 config', 'Value', '']

        table = ax.table(cellText=tb_list, colWidths=widths, colLabels=labels, cellLoc='left', loc='center')
        table.scale(0.8, 0.8)
        table.auto_set_font_size(False)

        for key, cell in table.get_celld().items():
            cell.set_fontsize(3.5)
            row, col = key
            if row == 0:
                cell.set_color('#ffb300')
                cell.set_fontsize(5)
            if col in [2, 5]:
                cell.set_color('white')
            if col in [1, 4, 7, 10, 13]:
                cell._loc = 'center'

        self._save_plots(fig, suffix='parameter_page')

    def _plot_occupancy(self, hist, electron_axis=False, use_electron_offset=False, title='Occupancy', z_label='# of hits', z_min=None, z_max=None, show_sum=True, suffix=None, extend_upper_bound=True, aspect='equal', colorbar_tick_count=10, colorbar_integer=False, colorbar_scientific=False):
        """Render a 2D map with optional electron axis and colorbar controls."""
        z_min, z_max = self._resolve_map_z_limits(hist, z_min=z_min, z_max=z_max)
        z_min, z_max = self._apply_map_range_override(suffix, z_min, z_max)

        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        self._add_text(fig)

        ax.set_adjustable('box')
        extent = self.plot_box_bounds
        bounds = np.linspace(start=z_min, stop=z_max + (1 if extend_upper_bound else 0), num=255, endpoint=True)
        cmap = copy.copy(cm.get_cmap('plasma'))
        cmap.set_bad('w')
        cmap.set_over('r')  # Make noisy pixels red
        cmap.set_under('g')
        norm = colors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(hist, interpolation='none', aspect=aspect, cmap=cmap, norm=norm, extent=extent)  # TODO: use pcolor or pcolormesh
        ax.set_ylim((self.plot_box_bounds[2], self.plot_box_bounds[3]))
        ax.set_xlim((self.plot_box_bounds[0], self.plot_box_bounds[1]))
        self._set_pixel_axis_ticks(ax)
        self._apply_axis_override(ax, suffix, allow_x=False, allow_y=False)
        if not show_sum:
            ax.set_title(title, color=TITLE_COLOR)
        else:
            ax.set_title(title + ' ($\\Sigma$ = {0})'.format((0 if hist.all() is np.ma.masked else np.ma.sum(hist))), color=TITLE_COLOR)

        if self._module_type is None or not self._module_type.switch_axis():
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
        else:
            ax.set_xlabel('Row')
            ax.set_ylabel('Column')

        divider = make_axes_locatable(ax)
        if colorbar_integer:
            ticks = self._integer_colorbar_ticks(z_min, z_max)
        else:
            ticks = np.linspace(start=z_min, stop=z_max + (1 if extend_upper_bound else 0), num=colorbar_tick_count, endpoint=True)
        if self.cb_side:  # and not electron_axis:
            pad = 0.8 if electron_axis else 0.2
            cax = divider.append_axes("right", size="5%", pad=pad)
            cb = fig.colorbar(im, cax=cax, ticks=ticks)
            axis = cb.ax.yaxis
        else:
            pad = 1.0 if electron_axis else 0.6
            cax = divider.append_axes("bottom", size="5%", pad=pad)
            cb = fig.colorbar(im, cax=cax, ticks=ticks, orientation='horizontal')
            axis = cb.ax.xaxis
        if colorbar_scientific:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            axis.set_major_formatter(formatter)
            cb.update_ticks()
        elif colorbar_integer:
            axis.set_major_formatter(FormatStrFormatter('%d'))
            cb.update_ticks()
        else:
            axis.set_major_formatter(FormatStrFormatter('%.1f'))
            cb.update_ticks()
        cb.set_label(z_label)

        if electron_axis:
            def f(x):
                return np.array([self._convert_to_e(x, use_offset=use_electron_offset)[0] for x in x])

            if self.cb_side:
                ax2 = cb.ax.secondary_yaxis('left', functions=(lambda x: x, lambda x: x))
                e_ax = ax2.yaxis
            else:
                ax2 = cb.ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
                e_ax = ax2.xaxis
            e_ax.set_ticks(ticks)
            e_ax.set_ticklabels(f(ticks).round().astype(int))
            e_ax.set_label_text('{0} [Electrons]'.format(z_label))
            axis.set_major_formatter(FormatStrFormatter('%.2f'))
            cb.update_ticks()
            cb.set_label('{0} [$\\Delta$ VCAL]'.format(z_label))

        self._save_plots(fig, suffix=suffix)

    def _plot_2d_param_hist(self, hist, scan_parameters, y_max=None, electron_axis=False, scan_parameter_name=None, title='Scan Parameter Histogram', ylabel='', suffix=None):
        """Render a scan-parameter heatmap such as ToT-vs-VCAL."""

        if y_max is None:
            y_max = hist.shape[0]

        x_bins = scan_parameters[:]  # np.arange(-0.5, max(scan_parameters) + 1.5)
        y_bins = np.arange(-0.5, y_max + 0.5)

        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        self._add_text(fig)

        fig.patch.set_facecolor('white')

        # log scale
        norm = colors.LogNorm()

        im = ax.pcolormesh(x_bins, y_bins, hist, norm=norm,shading='auto', rasterized=True)
        ax.set_xlim(x_bins[0], x_bins[-1])
        # ax.set_xlim(x_bins[0], 200)
        # ax.set_ylim(-0.5, y_max)
        ax.set_ylim(-0.5, 20)
        self._apply_axis_override(ax, suffix)

        cb = fig.colorbar(im, fraction=0.04, pad=0.05)

        cb.set_label("# of pixels")
        ax.set_title(title + " for {0} pixel(s)".format(self.n_enabled_pixels), color=TITLE_COLOR)
        if scan_parameter_name is None:
            ax.set_xlabel('Scan parameter')
        else:
            ax.set_xlabel(scan_parameter_name)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if electron_axis:
            self._add_electron_axis(fig, ax)

        self._save_plots(fig, suffix=suffix or 'histogram')

    def _plot_fancy_occupancy(self, hist, title='Occupancy', z_label='#', z_min=None, z_max=None, log_z=True, norm_projection=False, show_sum=True, centered_ticks=False, suffix='fancy_occupancy'):
        """Render a map together with row/column projections."""
        if log_z:
            title += '\n(logarithmic scale)'
        title += '\nwith projections'

        if z_min is None:
            z_min = np.ma.min(hist)
        if log_z and z_min == 0:
            z_min = 0.1
        if z_max is None:
            z_max = np.ma.max(hist)
        z_min, z_max = self._apply_map_range_override(suffix, z_min, z_max)

        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        self._add_text(fig)

        extent = self.plot_box_bounds
        if log_z:
            bounds = np.logspace(start=np.log10(z_min), stop=np.log10(z_max), num=255, endpoint=True)
        else:
            bounds = np.linspace(start=z_min, stop=z_max, num=int(z_max + 1), endpoint=True)
        if centered_ticks:
            cmap = copy.copy(cm.get_cmap('plasma', (z_max)))
        else:
            cmap = copy.copy(cm.get_cmap('plasma'))
        cmap.set_bad('w')
        norm = colors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(hist, interpolation='none', aspect='auto', cmap=cmap, norm=norm, extent=extent)  # TODO: use pcolor or pcolormesh
        ax.set_ylim((self.plot_box_bounds[2], self.plot_box_bounds[3]))
        ax.set_xlim((self.plot_box_bounds[0], self.plot_box_bounds[1]))
        self._set_pixel_axis_ticks(ax)
        self._apply_axis_override(ax, suffix, allow_x=False, allow_y=False)
        if self._module_type is None or not self._module_type.switch_axis():
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
        else:
            ax.set_xlabel('Row')
            ax.set_ylabel('Column')

        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(ax)
        axHistx = divider.append_axes("top", 1.2, pad=0.2, sharex=ax)
        axHisty = divider.append_axes("right", 1.2, pad=0.2, sharey=ax)

        cax = divider.append_axes("right", size="5%", pad=0.1)
        if log_z:
            cb = fig.colorbar(im, cax=cax, ticks=np.logspace(start=np.log10(z_min), stop=np.log10(z_max), num=9, endpoint=True))
        elif centered_ticks:
            ctick_size = (z_max - z_min) / (z_max)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=z_min - ctick_size / 2, vmax=z_max + ctick_size / 2))
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax, ticks=np.linspace(start=z_min, stop=z_max, num=int((z_max - z_min) + 1), endpoint=True))
        else:
            cb = fig.colorbar(im, cax=cax, ticks=np.linspace(start=z_min, stop=z_max, num=int((z_max - z_min) + 1), endpoint=True))
        cb.set_label(z_label)
        # make some labels invisible
        setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(), visible=False)
        if norm_projection:
            hight = np.ma.mean(hist, axis=0)
        else:
            hight = np.ma.sum(hist, axis=0)
        hight = self._filled_projection(hight)

        x_positions = np.linspace(self.plot_box_bounds[0] + 0.5, self.plot_box_bounds[1] - 0.5, hist.shape[1])
        axHistx.bar(x=x_positions, height=hight, align='center', linewidth=0)
        axHistx.set_xlim((self.plot_box_bounds[0], self.plot_box_bounds[1]))
        if hist.all() is np.ma.masked:
            axHistx.set_ylim((0, 1))
        axHistx.locator_params(axis='y', nbins=3)
        axHistx.ticklabel_format(style='sci', scilimits=(0, 4), axis='y')
        axHistx.set_ylabel(z_label)
        if norm_projection:
            width = np.ma.mean(hist, axis=1)
        else:
            width = np.ma.sum(hist, axis=1)
        width = self._filled_projection(width)

        y_positions = np.linspace(min(self.plot_box_bounds[2], self.plot_box_bounds[3]) + 0.5,
                                  max(self.plot_box_bounds[2], self.plot_box_bounds[3]) - 0.5,
                                  hist.shape[0])
        axHisty.barh(y=y_positions, width=width, align='center', linewidth=0)
        axHisty.set_ylim((self.plot_box_bounds[2], self.plot_box_bounds[3]))
        if hist.all() is np.ma.masked:
            axHisty.set_xlim((0, 1))
        axHisty.locator_params(axis='x', nbins=3)
        axHisty.ticklabel_format(style='sci', scilimits=(0, 4), axis='x')
        axHisty.set_xlabel(z_label)

        if not show_sum:
            ax.set_title(title, color=TITLE_COLOR, x=1.35, y=1.2)
        else:
            ax.set_title(title + '\n($\\Sigma$ = {0})'.format((0 if hist.all() is np.ma.masked else np.ma.sum(hist))), color=TITLE_COLOR, x=1.35, y=1.2)

        self._save_plots(fig, suffix=suffix)

    def _plot_scurves(self, scurves, scan_parameters, electron_axis=False, scan_parameter_name=None, suffix='scurves', title='S-curves', ylabel='Occupancy'):
        """Render the occupancy evolution over the scan parameter."""
        n_injections = self.scan_config.get('n_injections', 100)
        y_max = int(n_injections * 1.5)
        max_occ = y_max + 2
        if self.run_config['scan_id'] == 'autorange_threshold_scan':
            max_occ = int(min(np.max(scurves) + 5, y_max + 2))
        x_bins = scan_parameters  # np.arange(-0.5, max(scan_parameters) + 1.5)
        y_bins = np.arange(-0.5, max_occ + 0.5)

        noisy_mask = np.any(scurves > y_max, axis=0).reshape((self.cols, self.rows))
        n_noisy_pixels = np.count_nonzero(noisy_mask & ~self.enable_mask)

        param_count = scurves.shape[0]
        hist = np.empty([param_count, max_occ], dtype=np.uint32)
        scurves_clipped = np.clip(scurves, 0, max_occ - 1)
        enabled_pixels = ~self.enable_mask.reshape((scurves.shape[-1]))

        for param in range(param_count):
            if self.run_config['scan_id'] == 'autorange_threshold_scan':
                hist[param] = np.bincount(scurves_clipped[param, enabled_pixels].astype(int), minlength=max_occ)
            else:
                hist[param] = np.bincount(scurves_clipped[param, enabled_pixels], minlength=max_occ)

        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        self._add_text(fig)

        fig.patch.set_facecolor('white')
        cmap = copy.copy(cm.get_cmap('cool'))
        if np.allclose(hist, 0.0) or hist.max() <= 1:
            z_max = 1.0
        else:
            z_max = hist.max()
        # for small z use linear scale, otherwise log scale
        if z_max <= 10.0:
            bounds = np.linspace(start=0.0, stop=z_max, num=255, endpoint=True)
            norm = colors.BoundaryNorm(bounds, cmap.N)
        else:
            bounds = np.linspace(start=1.0, stop=z_max, num=255, endpoint=True)
            norm = colors.LogNorm()

        im = ax.pcolormesh(x_bins, y_bins, hist.T, norm=norm, rasterized=True, shading='flat')
        ax.set_ylim(-0.5, y_max)
        self._apply_axis_override(ax, suffix)

        if z_max <= 10.0:
            cb = fig.colorbar(im, ticks=np.linspace(start=0.0, stop=z_max, num=min(
                11, math.ceil(z_max) + 1), endpoint=True), fraction=0.04, pad=0.05)
        else:
            cb = fig.colorbar(im, fraction=0.04, pad=0.05)
        cb.set_label("# of pixels")
        ax.set_title(title + ' for {0} pixel(s)'.format(self.n_enabled_pixels), color=TITLE_COLOR)
        if scan_parameter_name is None:
            ax.set_xlabel('Scan parameter')
        else:
            ax.set_xlabel(scan_parameter_name)
        ax.set_ylabel(ylabel)

        text = 'Failed fits: {0}\nNoisy pixels: {1}'.format(self.n_failed_scurves, n_noisy_pixels)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.88, text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', bbox=props)

        if electron_axis:
            self._add_electron_axis(fig, ax)

        self._save_plots(fig, suffix=suffix)

    def _plot_stacked_threshold(self, data, tdac_mask, plot_range=None, electron_axis=False, x_axis_title=None, y_axis_title='# of hits', z_axis_title='TDAC',
                                title=None, suffix=None, min_tdac=15, max_tdac=0, range_tdac=16,
                                fit_gauss=True, plot_legend=True, centered_ticks=False, print_failed_fits=False):
        """Render the threshold distribution split by TDAC value."""
        flat_data = self._sanitize_distribution_data(data)
        if flat_data.size == 0:
            self.log.warning('No valid data available for stacked threshold plot.')
            return

        if plot_range is None:
            diff = np.amax(flat_data) - np.amin(flat_data)
            if np.amax(flat_data) > np.median(flat_data) * 5:
                plot_range = np.arange(
                    np.amin(flat_data), np.median(flat_data) * 5, diff / 100.)
            else:
                plot_range = np.arange(np.amin(flat_data), np.amax(flat_data) + diff / 100., diff / 100.)

        tick_size = plot_range[1] - plot_range[0]

        hist, bins = np.histogram(flat_data, bins=plot_range)

        bin_centres = (bins[:-1] + bins[1:]) / 2
        p0 = (np.amax(hist), np.nanmean(bins), (max(plot_range) - min(plot_range)) / 3)

        if fit_gauss:
            try:
                coeff, _ = curve_fit(au.gauss, bin_centres, hist, p0=p0)
            except Exception:
                coeff = None
                self.log.warning('Gauss fit failed!')
        else:
            coeff = None

        if coeff is not None:
            points = np.linspace(min(plot_range), max(plot_range), 500)
            gau = au.gauss(points, *coeff)

        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        self._add_text(fig)

        cmap = copy.copy(cm.get_cmap('viridis', (range_tdac)))
        # create dicts for tdac data
        data_thres_tdac = {}
        hist_tdac = {}
        tdac_bar = {}

        # select threshold data for different tdac values according to tdac map
        for tdac in range(range_tdac + 1):
            data_thres_tdac[tdac] = self._sanitize_distribution_data(data[tdac_mask == tdac])
            # histogram threshold data for each tdac
            hist_tdac[tdac], _ = np.histogram(data_thres_tdac[tdac], bins=bins)
            tdac_bar[tdac] = ax.bar(bins[:-1], hist_tdac[tdac], bottom=np.sum([hist_tdac[i] for i in range(tdac)], axis=0), width=tick_size, align='edge', color=cmap(1. / (range_tdac+1) * tdac), linewidth=0)

        fig.subplots_adjust(right=0.85)
        cax = fig.add_axes([0.89, 0.11, 0.02, 0.645])
        if centered_ticks:
            ctick_size = (max_tdac - min_tdac) / (range_tdac)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=min_tdac - ctick_size / 2, vmax=max_tdac + ctick_size / 2))
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=min_tdac, vmax=max_tdac))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, ticks=np.linspace(start=min_tdac, stop=max_tdac, num=range_tdac, endpoint=True))
        cb.set_label(z_axis_title)

        if coeff is not None:
            ax.plot(points, gau, "r-", label='Normal distribution')

        ax.set_xlim((min(plot_range), max(plot_range)))
        ax.set_title(title, color=TITLE_COLOR)
        if x_axis_title is not None:
            ax.set_xlabel(x_axis_title)
        if y_axis_title is not None:
            ax.set_ylabel(y_axis_title)
        ax.grid(True)
        self._apply_axis_override(ax, suffix)


        if plot_legend:
            stats_data = flat_data[flat_data < 1e5]
            if stats_data.size == 0:
                stats_data = flat_data
            mean = np.nanmean(stats_data)
            rms = np.nanstd(stats_data)
            if electron_axis:
                textright = '$\\mu={0:1.2f}\\;\\Delta$VCAL\n$\\;\\;\\,=({1[0]:1.0f} \\pm {1[1]:1.0f}) \\; e^-$\n\n$\\sigma={2:1.2f}\\;\\Delta$VCAL\n$\\;\\;\\,=({3[0]:1.0f} \\pm {3[1]:1.0f}) \\; e^-$'.format(mean, self._convert_to_e(mean), rms, self._convert_to_e(rms, use_offset=False))
            else:
                textright = '$\\mu={0:1.2f}\\;\\Delta$VCAL\n$\\sigma={1:1.2f}\\;\\Delta$VCAL'.format(mean, rms)

            # Fit results
            if coeff is not None:
                textright += '\n\nFit results:\n'
                if electron_axis:
                    textright += '$\\mu={0:1.2f}\\;\\Delta$VCAL\n$\\;\\;\\,=({1[0]:1.0f} \\pm {1[1]:1.0f}) \\; e^-$\n\n$\\sigma={2:1.2f}\\;\\Delta$VCAL\n$\\;\\;\\,=({3[0]:1.0f} \\pm {3[1]:1.0f}) \\; e^-$'.format(abs(coeff[1]), self._convert_to_e(abs(coeff[1])), abs(coeff[2]), self._convert_to_e(abs(coeff[2]), use_offset=False))
                else:
                    textright += '$\\mu={0:1.2f}\\;\\Delta$VCAL\n$\\sigma={1:1.2f}\\;\\Delta$VCAL'.format(abs(coeff[1]), abs(coeff[2]))

                textright += '\n\nFailed fits: {0}'.format(self.n_failed_scurves)
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.03, 0.95, textright, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

        if electron_axis:
            self._add_electron_axis(fig, ax)

        self._save_plots(fig, suffix=suffix)

    def _plot_distribution(self, data, plot_range=None, x_axis_title=None, electron_axis=False, use_electron_offset=False, y_axis_title='# of hits', log_y=False, align='edge', title=None, print_failed_fits=False, fit_gauss=True, plot_legend=True, suffix=None):
        """Render a 1D histogram with optional Gaussian summary."""
        data = self._sanitize_distribution_data(data)
        if data.size == 0:
            self.log.warning('No valid data available for distribution plot %s.', suffix if suffix is not None else '')
            return

        if plot_range is None:
            diff = np.amax(data) - np.amin(data)
            median = np.ma.median(data)
            if (np.amax(data)) > median * 5:
                plot_range = np.arange(np.amin(data), median * 2, median / 100.)
            else:
                plot_range = np.arange(np.amin(data), np.amax(data) + diff / 100., diff / 100.)
        tick_size = np.diff(plot_range)[0]

        hist, bins = np.histogram(np.ravel(data), bins=plot_range)

        bin_centres = (bins[:-1] + bins[1:]) / 2
        p0 = (np.amax(hist), np.nanmean(bins), (max(plot_range) - min(plot_range)) / 3)

        if fit_gauss:
            try:
                coeff, _ = curve_fit(au.gauss, bin_centres, hist, p0=p0)
            except Exception:
                coeff = None
                self.log.warning('Gauss fit failed!')
        else:
            coeff = None

        if coeff is not None:
            points = np.linspace(min(plot_range), max(plot_range), 500)
            gau = au.gauss(points, *coeff)

        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        self._add_text(fig)

        ax.bar(bins[:-1], hist, width=tick_size, align=align)
        if coeff is not None:
            ax.plot(points, gau, "r-", label='Normal distribution')

        if log_y:
            if title is not None:
                title += ' (logscale)'
            ax.set_yscale('log')

        # if log_x:
        #     if title is not None:
        #         title += ' (logscale)'
        #     ax.set_xscale('log')

        ax.set_xlim(min(plot_range), max(plot_range))
        ax.set_title(title, color=TITLE_COLOR)
        if x_axis_title is not None:
            ax.set_xlabel(x_axis_title)
        if y_axis_title is not None:
            ax.set_ylabel(y_axis_title)
        ax.grid(True)
        self._apply_axis_override(ax, suffix)

        if plot_legend:
            sel = (data < 1e5)
            mean = np.nanmean(data[sel])
            rms = np.nanstd(data[sel])
            if electron_axis:
                textright = '$\\mu={0:1.2f}\\;\\Delta$VCAL\n$\\;\\;\\,=({1[0]:1.0f} \\pm {1[1]:1.0f}) \\; e^-$\n\n$\\sigma={2:1.2f}\\;\\Delta$VCAL\n$\\;\\;\\,=({3[0]:1.0f} \\pm {3[1]:1.0f}) \\; e^-$'.format(mean, self._convert_to_e(mean, use_offset=use_electron_offset), rms, self._convert_to_e(rms, use_offset=False))
            else:
                textright = '$\\mu={0:1.2f}\\;\\Delta$VCAL\n$\\sigma={1:1.2f}\\;\\Delta$VCAL'.format(mean, rms)
            if print_failed_fits:
                textright += '\n\nFailed fits: {0}'.format(self.n_failed_scurves)

            # Fit results
            if coeff is not None:
                textright += '\n\nFit results:\n'
                if electron_axis:
                    textright += '$\\mu={0:1.2f}\\;\\Delta$VCAL\n$\\;\\;\\,=({1[0]:1.0f} \\pm {1[1]:1.0f}) \\; e^-$\n\n$\\sigma={2:1.2f}\\;\\Delta$VCAL\n$\\;\\;\\,=({3[0]:1.0f} \\pm {3[1]:1.0f}) \\; e^-$'.format(abs(coeff[1]), self._convert_to_e(abs(coeff[1]), use_offset=use_electron_offset), abs(coeff[2]), self._convert_to_e(abs(coeff[2]), use_offset=False))
                else:
                    textright += '$\\mu={0:1.2f}\\;\\Delta$VCAL\n$\\sigma={1:1.2f}\\;\\Delta$VCAL'.format(abs(coeff[1]), abs(coeff[2]))
                if print_failed_fits:
                    textright += '\n\nFailed fits: {0}'.format(self.n_failed_scurves)

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.03, 0.95, textright, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

        if electron_axis:
            self._add_electron_axis(fig, ax, use_electron_offset=use_electron_offset)

        self._save_plots(fig, suffix=suffix)

    def _plot_1d_hist(self, hist, yerr=None, title=None, x_axis_title=None, y_axis_title=None, x_ticks=None, color='r',
                      plot_range=None, log_y=False, suffix=None, plot_legend=True):
        """Render a simple 1D histogram from pre-binned data."""
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        self._add_text(fig)

        hist = np.array(hist)
        if plot_range is None:
            plot_range = range(0, len(hist))
        plot_range = np.array(plot_range)
        plot_range = plot_range[plot_range < len(hist)]
        if yerr is not None:
            ax.bar(x=plot_range, height=hist[plot_range], color=color, align='center', yerr=yerr)
        else:
            ax.bar(x=plot_range, height=hist[plot_range], color=color, align='center')
        ax.set_xlim((min(plot_range) - 0.5, max(plot_range) + 0.5))

        ax.set_title(title, color=TITLE_COLOR)
        if x_axis_title is not None:
            ax.set_xlabel(x_axis_title)
        if y_axis_title is not None:
            ax.set_ylabel(y_axis_title)
        if x_ticks is not None:
            ax.set_xticks(plot_range)
            ax.set_xticklabels(x_ticks)
            ax.tick_params(which='both', labelsize=8)
        if np.allclose(hist, 0.0):
            ax.set_ylim((0, 1))
        else:
            if log_y:
                ax.set_yscale('log')
                ax.set_ylim((1e-1, np.amax(hist) * 2))
        ax.grid(True)
        self._apply_axis_override(ax, suffix)

        # if plot_legend:
        #     sel = (hist < 1e5)
        #     mean = np.nanmean(hist[sel])
        #     rms = np.nanstd(hist[sel])
        #     print('mean',mean)
        #     textright = '$\\mu={0:1.2f}\\;\\Delta$VCAL\n$\\sigma={1:1.2f}\\;\\Delta$VCAL'.format(mean, rms)

        self._save_plots(fig, suffix=suffix)

    def _plot_cl_shape(self, hist):
        ''' Create a histogram with selected cluster shapes '''
        x = np.arange(12)
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        self._add_text(fig)

        selected_clusters = hist[[1, 3, 5, 6, 9, 13, 14, 7, 11, 19, 261, 15]]
        ax.bar(x, selected_clusters, align='center')
        ax.xaxis.set_ticks(x)
        fig.subplots_adjust(bottom=0.2)
        ax.set_xticklabels(["\u2004\u2596",
                            # 2 hit cluster, horizontal
                            "\u2597\u2009\u2596",
                            # 2 hit cluster, vertical
                            "\u2004\u2596\n\u2004\u2598",
                            "\u259e",  # 2 hit cluster
                            "\u259a",  # 2 hit cluster
                            "\u2599",  # 3 hit cluster, L
                            "\u259f",  # 3 hit cluster
                            "\u259b",  # 3 hit cluster
                            "\u259c",  # 3 hit cluster
                            # 3 hit cluster, horizontal
                            "\u2004\u2596\u2596\u2596",
                            # 3 hit cluster, vertical
                            "\u2004\u2596\n\u2004\u2596\n\u2004\u2596",
                            # 4 hit cluster
                            "\u2597\u2009\u2596\n\u259d\u2009\u2598"])
        ax.set_title('Cluster shapes', color=TITLE_COLOR)
        ax.set_xlabel('Cluster shape')
        ax.set_ylabel('# of clusters')
        ax.grid(True)
        ax.set_yscale('log')
        ax.set_ylim(ymin=1e-1)

        self._save_plots(fig, suffix='cluster_shape')
