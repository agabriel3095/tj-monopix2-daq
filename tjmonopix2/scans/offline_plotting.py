"""Offline plotting entrypoint for the shared plotting module.

Main usage modes:
1. Pass one or more interpreted files on the command line.
2. Pass nothing and plot the latest ``*_interpreted.h5`` found in the
   default output directory.

By default, the produced PDF is named like:
``<scan_name>_interpreted_offline_plot.pdf``

If that PDF already exists, plotting is skipped unless ``-P`` is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tjmonopix2.analysis import plotting


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'output_data' / 'module_0' / 'chip_0'
MAP_PLOTS = {'occupancy_map', 'fancy_occupancy', 'tdac_map', 'tot_map', 'threshold_map', 'noise_map'}
HIST2D_PLOTS = {'tot_hist', 'scurves'}
LINE_PLOTS = {'monitoring_main', 'hit_pix', 'tdac_plot', 'tot_plot', 'threshold_plot', 'stacked_threshold', 'noise_plot', 'cluster_tot', 'cluster_size', 'tdc_status', 'dac_linearity'}
NO_AXIS_OVERRIDE_PLOTS = {'parameter_page', 'monitoring_summary', 'cluster_shape'}

# Main offline plotting defaults.
# Edit this block for your usual workflow instead of passing many CLI flags.
#
# Config keys:
# - exclude_plots:
#   Plot ids to remove from the default "all available plots" selection.
# - axis_ranges:
#   Per-plot axis overrides. Supported keys are x, y, z.
#   Axis support by plot type:
#   z     : 2D maps like occupancy_map, tdac_map, tot_map, threshold_map, noise_map
#   x,y   : 1D plots like hit_pix, tdac_plot, tot_plot, threshold_plot,
#           stacked_threshold, noise_plot, cluster_tot, cluster_size, tdc_status
#   x,y   : 2D histograms like tot_hist and scurves
#   none  : parameter_page, monitoring_summary
#   x,y   : monitoring_main and monitoring time-series style plots
# - map_splits:
#   Optional zoom tiling for selected 2D maps inside the scanned area.
#   Use a per-plot dictionary, for example:
#   {'noise_map': {'rows': 6, 'cols': 1}, 'threshold_map': {'rows': 3, 'cols': 2}}
#   Full matrix and scan-area plots stay unchanged; extra zoomed map tiles are
#   added only when:
#   scan_rows % rows == 0, scan_cols % cols == 0,
#   scan_rows / rows > 2, scan_cols / cols > 2
# - map_views:
#   Control whether 2D maps are shown on the full matrix and/or scan area.
#   Typical offline choice:
#   {'full_matrix': False, 'scan_area': True}
# - list_plots:
#   Print available plot ids and exit.
#
# Example:
# plot_config = {
#     'exclude_plots': ['fancy_occupancy'],
#     'axis_ranges': {
#         'threshold_plot': {'x': (0, 80)},
#         'noise_map': {'z': (0, 15)},
#     },
#     'map_splits': {
#         'noise_map': {'rows': 6, 'cols': 1},
#     },
#     'map_views': {'full_matrix': False, 'scan_area': True},
#     'list_plots': False,
# }
plot_config = {
    'exclude_plots': [],
    'axis_ranges': {},
    'map_splits': None,
    'map_views': {'full_matrix': False, 'scan_area': True},
    'list_plots': False,
}


def _parse_plot_list(values):
    """Flatten repeated or comma-separated plot options."""
    result = []
    for value in values or []:
        for item in value.split(','):
            item = item.strip()
            if item:
                result.append(item)
    return result


def _parse_pair(value):
    """Parse a numeric min,max pair."""
    low, high = value.split(',', 1)
    return float(low), float(high)


def _parse_axis_ranges(values):
    """Parse axis overrides of the form plot_id:x=min,max:y=min,max:z=min,max."""
    ranges = {}
    for value in values or []:
        parts = value.split(':')
        plot_id = parts[0].strip()
        if not plot_id:
            raise ValueError(f'Invalid axis range specification: {value}')
        ranges.setdefault(plot_id, {})
        for part in parts[1:]:
            key, raw = part.split('=', 1)
            key = key.strip()
            if key not in {'x', 'y', 'z'}:
                raise ValueError(f'Unknown axis selector "{key}" in: {value}')
            ranges[plot_id][key] = _parse_pair(raw.strip())
    return ranges


def _build_selected_plots(include, exclude):
    """Resolve the final selected plot list."""
    include_plots = _parse_plot_list(include)
    exclude_plots = set(_parse_plot_list(exclude))
    if include_plots:
        return [plot_id for plot_id in include_plots if plot_id not in exclude_plots]
    return None


def _merge_with_plot_config(args):
    """Merge CLI arguments with the script-level plot configuration."""
    config = dict(plot_config)
    config['exclude_plots'] = list(config.get('exclude_plots', []))
    config['axis_ranges'] = dict(config.get('axis_ranges', {}))
    config['map_splits'] = dict(config['map_splits']) if config.get('map_splits') else None
    config['map_views'] = dict(config.get('map_views', {}))

    cli_exclude_plots = _parse_plot_list(args.exclude_plot)
    if cli_exclude_plots:
        config['exclude_plots'] = cli_exclude_plots

    cli_axis_ranges = _parse_axis_ranges(args.axis_range)
    if cli_axis_ranges:
        config['axis_ranges'] = cli_axis_ranges

    if args.list_plots:
        config['list_plots'] = True

    return config


def _find_latest_interpreted_file(directory):
    """Return the newest interpreted file in a directory."""
    candidates = sorted(directory.glob('*_interpreted.h5'), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f'No *_interpreted.h5 files found in {directory}')
    return candidates[-1]


def _resolve_input_files(cli_files, directory):
    """Resolve input files from CLI or fall back to the latest interpreted file."""
    if cli_files:
        return [Path(file_name).resolve() for file_name in cli_files]
    return [_find_latest_interpreted_file(directory.resolve())]


def _default_pdf_path(input_file):
    """Return the default offline PDF path for an interpreted file."""
    return input_file.with_name(f'{input_file.stem}_offline_plot.pdf')


def _resolve_pdf_path(input_file):
    """Resolve the output PDF file path for a given interpreted file."""
    return _default_pdf_path(input_file)


def _should_skip_plotting(pdf_path, force):
    """Return True when plotting should be skipped."""
    return pdf_path.exists() and not force


def _axis_override_description(plot_id):
    """Return a short description of supported axis overrides for a plot."""
    if plot_id in MAP_PLOTS:
        return 'z + optional zoom tiles'
    if plot_id in HIST2D_PLOTS:
        return 'x,y'
    if plot_id in LINE_PLOTS:
        return 'x,y'
    if plot_id in NO_AXIS_OVERRIDE_PLOTS:
        return 'none'
    return 'x,y'


def main():
    """Run shared plotting on one or more interpreted files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', help='Interpreted HDF5 files to plot. If omitted, use the latest one from the configured directory.')
    parser.add_argument('-d', '--directory', default=None, help='Directory used to find the latest interpreted file when no file is passed.')
    parser.add_argument('--exclude-plot', action='append', default=[], help='Plot id to exclude. Can be repeated or comma-separated.')
    parser.add_argument('--axis-range', action='append', default=[], help='Axis override like threshold_map:x=350,380:y=0,512:z=0,30')
    parser.add_argument('--list-plots', action='store_true', help='List available plot ids for the selected file and exit.')
    parser.add_argument('-P', dest='force_plotting', action='store_true', help='Force plotting even if the offline PDF already exists. If no interpreted.h5 is specified, it takes the last one in the folder.')
    args = parser.parse_args()

    config = _merge_with_plot_config(args)
    selected_plots = _build_selected_plots(None, config['exclude_plots'])
    input_directory = Path(args.directory).resolve() if args.directory else DEFAULT_OUTPUT_DIR
    input_files = _resolve_input_files(args.files, input_directory)

    first_file = input_files[0]
    with plotting.Plotting(
        analyzed_data_file=str(first_file),
        pdf_file=str(_resolve_pdf_path(first_file)),
        notify=False,
        show_progress=True,
        axis_ranges=config['axis_ranges'],
        map_split_config=config['map_splits'],
        map_output_config=config['map_views'],
        create_output=False,
    ) as plotter:
        if config['list_plots']:
            print('Available plots:')
            print('  plot id              label                           axis override')
            for plot_id, label in plotter.list_available_plot_details():
                axis_info = _axis_override_description(plot_id)
                print(f'  {plot_id:<20} {label:<30} {axis_info}')
            return

    for input_file in input_files:
        pdf_path = _resolve_pdf_path(input_file)
        if _should_skip_plotting(pdf_path, args.force_plotting):
            print(f'Skipping plotting for {input_file.name}: {pdf_path.name} already exists. Use -P to force plotting.')
            continue

        print(f'Plotting: {input_file.name}')
        with plotting.Plotting(
            analyzed_data_file=str(input_file),
            pdf_file=str(pdf_path),
            notify=False,
            show_progress=True,
            axis_ranges=config['axis_ranges'],
            map_split_config=config['map_splits'],
            map_output_config=config['map_views'],
        ) as plotter:
            plotter.create_selected_plots(selected_plots=selected_plots)


if __name__ == '__main__':
    main()
