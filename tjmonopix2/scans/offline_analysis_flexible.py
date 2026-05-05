"""Offline analysis helper with flexible raw/interpreted file handling.

This script is a safer replacement for the older offline analysis helpers.
The main improvement is that it no longer relies on hard-coded filename
patterns such as ``*164304*.h5`` to find raw files.

Supported workflows:
1. Interpret raw files (`-i` / `-I`)
2. Plot interpreted files (`-p` / `-P`)
3. Pass explicit raw or interpreted files on the command line
4. If no input is given, search inside ``-d`` for matching files
"""

from __future__ import annotations

import argparse
import glob
import os
import os.path as path
import shutil
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import tables as tb
import yaml
from tables import NoSuchNodeError

from tjmonopix2.analysis import analysis, plotting


def find_latest_file(directory: str, partial_name: str) -> str | None:
    """Return the most recent file in a directory whose name contains a token."""
    files = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if partial_name in name and os.path.isfile(os.path.join(directory, name))
    ]
    if not files:
        return None
    return max(files, key=os.path.getctime)


def is_interpreted_file(file_path: str) -> bool:
    """Check whether a file name already points to an interpreted scan file."""
    return file_path.endswith('_interpreted.h5')


def interpreted_path_from_raw(raw_file: str) -> str:
    """Return the interpreted output path that analysis would create for a raw file."""
    return raw_file.rsplit('.h5', 1)[0] + '_interpreted.h5'


def list_raw_h5_files(directory: str) -> List[str]:
    """List raw HDF5 files in a directory, excluding already interpreted outputs."""
    return sorted(
        file_path
        for file_path in glob.glob(os.path.join(directory, '*.h5'))
        if not is_interpreted_file(file_path)
    )


def list_interpreted_h5_files(directory: str) -> List[str]:
    """List interpreted HDF5 files in a directory."""
    return sorted(glob.glob(os.path.join(directory, '*_interpreted.h5')))


def calculate_mean_tot_map(hist_tot):
    """Build a mean-ToT map pixel by pixel from the HistTot histogram."""
    mean_tot = np.zeros((512, 512))
    bins = np.linspace(1, 127, 128)

    for col in range(512):
        for row in range(512):
            heights = hist_tot[col][row][0]
            if np.sum(heights) > 0:
                mean_tot[col][row] = np.dot(bins, heights) / np.sum(heights)
            else:
                mean_tot[col][row] = 0
    return mean_tot


def plot_pixmap_generic(map_data, mask_out, props, basename, output_dir):
    """Plot a generic 2D map, optionally restricted to the scanned area."""
    run_config = props['run_config']
    scan_config = props['scan_config']
    show_area_only = props.get('show_area_only', True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    map_data = np.array(map_data, copy=True)
    map_data[mask_out] = float('nan')
    image = plt.imshow(np.transpose(map_data), aspect='auto', interpolation='none')

    ax.set_xlabel('column')
    ax.set_ylabel('row')

    if show_area_only:
        ax.set_xlim((float(scan_config['start_column']) - 0.5, float(scan_config['stop_column']) - 0.5))
        ax.set_ylim((float(scan_config['stop_row']) - 0.5, float(scan_config['start_row']) - 0.5))

    cbar = plt.colorbar(image)
    cbar.set_label(props.get('colorbar_label', ''))

    plt.title(run_config['chip_sn'] + ': ' + props.get('title', ''))

    masked_str = '\nnoisy pixels: {}'.format(np.sum(mask_out, axis=(0, 1)))
    plt.text(
        0,
        -0.1,
        'Scan id: ' + run_config['scan_id'] + masked_str,
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, basename + '_hitmap_' + props.get('output-name', 'output_name_undefined') + '.png'))
    plt.close(fig)


def plot_tot_histograms(hist_tot, mask_out, props, basename, output_dir):
    """Plot frontend-separated ToT histograms."""
    run_config = props['run_config']

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    hist_tot = np.array(hist_tot, copy=True)
    hist_tot[mask_out, :] = 0
    bins = np.linspace(1, 127, 128)
    boundaries = [0, 224, 448, 480, 512]
    labels = ['Normal FE', 'Normal casc. FE', 'HV casc. FE', 'HV FE']

    for i in range(4):
        hist = np.sum(hist_tot[boundaries[i]:boundaries[i + 1], :, :], axis=(0, 1, 2))
        ax.plot(bins, hist, label=labels[i])

    plt.xlabel('ToT / 25ns')
    plt.ylabel('Occurances')
    plt.title(run_config['chip_sn'] + ': ToT Histogram')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(output_dir, basename + '_hitmap_hist_tot.png'))
    plt.close(fig)


def plot_occ_histograms(map_data, mask_out, props, basename, output_dir):
    """Plot occupancy histograms frontend by frontend."""
    run_config = props['run_config']

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    map_data = np.array(map_data, copy=True)
    map_data[mask_out] = 0

    boundaries = [0, 224, 448, 480, 512]
    labels = ['Normal FE', 'Normal casc. FE', 'HV casc. FE', 'HV FE']

    frontend_data = []
    max_values = []
    for i in range(4):
        values = map_data[boundaries[i]:boundaries[i + 1], :].reshape(-1)
        values = values[values != 0]
        frontend_data.append(values)
        max_values.append(np.amax(values, initial=0))

    for i in range(4):
        ax.hist(
            frontend_data[i],
            rwidth=0.9,
            label=labels[i],
            alpha=0.6,
            edgecolor='black',
            bins=20,
            range=(0, np.amax(max_values)),
        )

    plt.xlabel('Number of Hits')
    plt.ylabel('Number of Pixels')
    plt.title(run_config['chip_sn'] + ': Hit Histogram')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, basename + '_hitmap_hist_occ.png'))
    plt.close(fig)


def export_mask_yaml(path_h5, basepath, noisy_pixels, occ, clim, measurement):
    """Export masked/noisy pixels to YAML for later reuse."""
    masked_pixels = []

    with tb.open_file(path_h5, 'r') as in_file:
        # Interpreted files always contain configuration_in.
        # Some of them also keep configuration_out, but not all scan types do.
        if hasattr(in_file.root, 'configuration_in') and hasattr(in_file.root.configuration_in.chip, 'use_pixel'):
            pixel_mask = in_file.root.configuration_in.chip.use_pixel[:]
            config_node_name = 'configuration_in'
        elif hasattr(in_file.root, 'configuration_out') and hasattr(in_file.root.configuration_out.chip, 'use_pixel'):
            pixel_mask = in_file.root.configuration_out.chip.use_pixel[:]
            config_node_name = 'configuration_out'
        else:
            raise NoSuchNodeError('Could not find chip.use_pixel in configuration_in or configuration_out')

    disabled_pixels = np.array(np.where(pixel_mask == False))

    print(f'--- Masked pixel from {config_node_name}.chip.use_pixel in ---', path_h5)
    print('[row,col]')
    for i in range(np.shape(disabled_pixels)[1]):
        row = disabled_pixels[1, i]
        col = disabled_pixels[0, i]
        masked_pixels.append({'row': int(row), 'col': int(col), 'hits': 999999.0})

    for row in range(512):
        for col in range(512):
            if noisy_pixels[col, row]:
                masked_pixels.append({'row': row, 'col': col, 'hits': float(occ[col, row])})

    output = {
        'measurement': measurement,
        'median_hits': float(np.median(occ[occ > 0])),
        'std_hits': float(np.std(occ[occ > 0])),
        'cutoff': float(clim),
        'masked_pixels': masked_pixels,
    }
    with open(path.join(basepath, 'masked_pixels.yaml'), 'w') as outfile:
        yaml.dump(output, outfile, default_flow_style=False, sort_keys=False)


def table_to_dict(table_item, key_name='attribute', value_name='value'):
    """Convert a PyTables config table into a plain Python dictionary."""
    ret = {}
    for row in table_item.iterrows():
        ret[row[key_name].decode('UTF-8')] = row[value_name].decode('UTF-8')
    return ret


def prepare_output_directory(path_h5, force=True):
    """Create or refresh the directory where PNG plots are written."""
    output_dir = path_h5.rsplit('_interpreted.h5')[0]
    if path.isdir(output_dir):
        if force:
            for file_path in glob.glob(os.path.join(output_dir, '*')):
                os.remove(file_path)
        else:
            return None
    else:
        os.mkdir(output_dir)
    return output_dir


def plot_from_file(path_h5, output_dir, clim):
    """Create the custom PNG summary plots from an interpreted HDF5 file."""
    if output_dir is None:
        return
    basename = path.basename(output_dir)
    try:
        h5file = tb.open_file(path_h5, mode='r', title='configuration_in')

        hist_occ = np.asarray(h5file.root.HistOcc)[:, :, 0].astype(float)
        hist_occ_original = hist_occ.copy()
        hist_tot = np.asarray(h5file.root.HistTot).astype(float)
        avg_tot = calculate_mean_tot_map(hist_tot)

        scan_config = table_to_dict(h5file.root.configuration_in.scan.scan_config)
        run_config = table_to_dict(h5file.root.configuration_in.scan.run_config)
    except NoSuchNodeError:
        print('error in reading h5 file: probably not complete')
        return
    finally:
        h5file.close()

    selector = np.zeros(hist_occ.shape, dtype=bool)
    selector[
        int(scan_config['start_column']):int(scan_config['stop_column']),
        int(scan_config['start_row']):int(scan_config['stop_row'])
    ] = True

    if clim == 'auto':
        if run_config['scan_id'] == 'analog_scan':
            clim = int(scan_config['n_injections'])
        else:
            clim = np.median(hist_occ[selector]) + (np.std(hist_occ[selector]) + 5) * 3
    elif clim != 'off':
        clim = int(clim)
    else:
        clim = None

    noisy_pixels = hist_occ > clim if clim else np.zeros_like(hist_occ, dtype=bool)

    prop_occ = {
        'colorbar_label': 'Occupancy',
        'title': 'Occupancy map',
        'output-name': 'occ',
        'run_config': run_config,
        'scan_config': scan_config,
        'show_area_only': True,
    }
    plot_pixmap_generic(hist_occ, noisy_pixels, prop_occ, basename, output_dir)
    prop_occ['show_area_only'] = False
    prop_occ['output-name'] = 'occ-full'
    plot_pixmap_generic(hist_occ, noisy_pixels, prop_occ, basename, output_dir)

    prop_tot = {
        'colorbar_label': 'Mean ToT / 25ns',
        'title': 'average ToT map',
        'output-name': 'tot',
        'run_config': run_config,
        'scan_config': scan_config,
    }
    plot_pixmap_generic(avg_tot, noisy_pixels, prop_tot, basename, output_dir)

    prop_hist = {
        'run_config': run_config,
        'scan_config': scan_config,
    }
    plot_tot_histograms(hist_tot, noisy_pixels, prop_hist, basename, output_dir)
    plot_occ_histograms(hist_occ, noisy_pixels, prop_hist, basename, output_dir)
    export_mask_yaml(path_h5, output_dir, noisy_pixels, hist_occ_original, clim, basename)


def interpret_raw_file(raw_file: str, force: bool) -> str | None:
    """Interpret one raw HDF5 file unless an interpreted copy already exists."""
    interpreted_file = interpreted_path_from_raw(raw_file)
    if path.isfile(interpreted_file):
        if force:
            os.remove(interpreted_file)
        else:
            print('Skipping already interpreted file:', path.basename(raw_file))
            return interpreted_file

    print('Analyzing file:', path.basename(raw_file))
    with analysis.Analysis(raw_data_file=raw_file, cluster_hits=True) as analyzer:
        analyzer.analyze_data()
    return interpreted_file if path.isfile(interpreted_file) else None


def resolve_raw_inputs(args) -> List[str]:
    """Resolve which raw files should be interpreted."""
    explicit_raw = [file_path for file_path in args.input_files if not is_interpreted_file(file_path)]
    if explicit_raw:
        return explicit_raw
    return list_raw_h5_files(args.d)


def resolve_interpreted_inputs(args) -> List[str]:
    """Resolve which interpreted files should be plotted."""
    explicit_inputs = list(args.input_files)
    if explicit_inputs:
        resolved = []
        for file_path in explicit_inputs:
            if is_interpreted_file(file_path):
                resolved.append(file_path)
            else:
                interpreted_file = interpreted_path_from_raw(file_path)
                if path.isfile(interpreted_file):
                    resolved.append(interpreted_file)
        return resolved

    latest = find_latest_file(args.d, '_scan_interpreted.h5')
    return [latest] if latest else []


def collect_plot_pngs(output_dir: str, collect_dir: str) -> None:
    """Copy generated PNGs into a common collection directory."""
    for file_path in glob.glob(os.path.join(output_dir, '*.png')):
        shutil.copy(file_path, path.join(collect_dir, path.basename(file_path)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default='./output_data/module_0/chip_0', help='directory to search for h5 files')
    parser.add_argument(
        'input_files',
        nargs='*',
        help='Optional raw or interpreted HDF5 file(s). If omitted, files are resolved from -d.',
    )
    parser.add_argument('-i', action='store_true', default=None, help='interpret raw h5 files that are not interpreted')
    parser.add_argument('-I', action='store_true', default=None, help='always re-interpret raw h5 files')
    parser.add_argument('-p', action='store_true', default=None, help='plot data from interpreted h5 files')
    parser.add_argument('-P', action='store_true', default=None, help='force replot of interpreted h5 files')
    parser.add_argument(
        '--clim',
        default='auto',
        help='limits of the colorbar for the hitmaps, either a number, auto, or off',
    )
    parser.add_argument(
        '--collect-plots',
        action='store_true',
        default=None,
        help='copy all generated PNGs into a single "plots" directory',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    collect_dir = os.path.join(args.d, 'plots')
    if args.collect_plots and not path.isdir(collect_dir):
        os.mkdir(collect_dir)

    interpreted_any = False
    if args.i or args.I:
        raw_files = resolve_raw_inputs(args)
        if not raw_files:
            print('No raw files found to interpret.')
        for raw_file in raw_files:
            interpreted_file = interpret_raw_file(raw_file, force=bool(args.I))
            interpreted_any = interpreted_any or interpreted_file is not None
        if not (args.p or args.P):
            print('[INFO] Interpretation finished. Use -p or -P as well if you also want plots in the same command.')

    if args.p or args.P:
        interpreted_files = resolve_interpreted_inputs(args)
        if not interpreted_files:
            print('No interpreted files found to plot.')
        for interpreted_file in interpreted_files:
            output_dir = prepare_output_directory(interpreted_file, force=bool(args.P))
            if output_dir:
                print('Plotting:', path.basename(interpreted_file))
                plot_from_file(interpreted_file, output_dir, args.clim)
                if args.collect_plots:
                    collect_plot_pngs(output_dir, collect_dir)
            with plotting.Plotting(analyzed_data_file=interpreted_file) as plotter:
                plotter.create_standard_plots()


if __name__ == '__main__':
    main()
