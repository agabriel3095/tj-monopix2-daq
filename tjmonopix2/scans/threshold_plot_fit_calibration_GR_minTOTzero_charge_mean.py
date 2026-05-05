import argparse
import os
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.backends.backend_pdf as pdf
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tables as tb
from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit

DEFAULT_PATH = '/home/labb2/tj-monopix2-daq-development/tjmonopix2/scans/output_data/module_0/chip_0/'
MAX_TOT_BIN_FOR_FIT = 120
THRESHOLD_FILTER_FRACTION = 0.05
TOT_BIN_ERROR = 1.0 / np.sqrt(12.0)
REGIONS = (
    (0, 224, 'NF'),
    (224, 448, 'NF CASC'),
    (448, 480, 'HV CASC'),
    (480, 512, 'HV'),
)


@dataclass(frozen=True)
class ScanConfig:
    start_column: int
    stop_column: int
    start_row: int
    stop_row: int
    n_injections: int
    vcal_low_start: int
    vcal_low_stop: int
    vcal_low_step: int
    vcal_high: int


@dataclass(frozen=True)
class Region:
    start_col: int
    stop_col: int
    start_row: int
    stop_row: int
    label: str
    frontend_start_col: int
    frontend_stop_col: int


@dataclass
class RegionCalibration:
    fit_params: np.ndarray
    chi2_ndof: float


class LogFormatterSciNotation(ticker.LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        return r'$10^{{{}}}$'.format(int(np.log10(x)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Unified threshold plot ToT calibration with 2-parameter or 3-parameter fit.'
    )
    parser.add_argument(
        '-f',
        '--input_file',
        type=str,
        default=None,
        help='Full path to the HDF5 file or a file name inside the default scan directory.'
    )
    parser.add_argument(
        '--fit_mode',
        choices=('fit2par', 'fit3par'),
        default='fit3par',
        help='Choose whether to run the historical 2-parameter or 3-parameter fit.'
    )
    parser.add_argument('--fit_range', type=int, default=0, help='Common fit start ToT bin for the tested frontend.')
    parser.add_argument('-plot', '--plot_range', type=int, default=0, help='Plot start bin.')
    parser.add_argument(
        '--max_tot_fit_bin',
        type=int,
        default=MAX_TOT_BIN_FOR_FIT,
        help='Largest ToT bin accepted in the charge-mean calibration.'
    )
    parser.add_argument(
        '--threshold_fraction',
        type=float,
        default=THRESHOLD_FILTER_FRACTION,
        help='Reject charge bins below this fraction of the row maximum before fitting.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed debugging information on the terminal.'
    )
    parser.add_argument(
        '--save_charge_debug_pdf',
        action='store_true',
        help='Save the charge-per-ToT Gaussian debug plots alongside the analyzed HDF5 file.'
    )
    return parser


def build_colormap() -> ListedColormap:
    viridis = colormaps.get_cmap('viridis').resampled(256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])
    return ListedColormap(newcolors)


def find_latest_file(directory: str, partial_name: str) -> str | None:
    files = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if partial_name in name and os.path.isfile(os.path.join(directory, name))
    ]
    if not files:
        return None
    return max(files, key=os.path.getctime)


def resolve_input_file(user_input: str | None) -> str:
    if user_input is None:
        latest = find_latest_file(DEFAULT_PATH, '_threshold_scan_interpreted.h5')
        if latest is None:
            print(f'[INFO] No interpreted HDF5 file found in {DEFAULT_PATH}.')
            sys.exit(0)
        print(f'[INFO] No file specified: using latest {latest}')
        return latest

    if os.sep not in user_input:
        candidate = os.path.join(DEFAULT_PATH, user_input)
    else:
        candidate = user_input

    if not os.path.exists(candidate):
        print(f'[ERROR] File not found: {candidate}')
        sys.exit(1)
    return candidate


def read_scan_config(file_path: str) -> ScanConfig:
    with tb.open_file(file_path, 'r') as h5_file:
        table = h5_file.root.configuration_in.scan.scan_config

        def read_int(attribute: str) -> int:
            raw = table.read_where(f'attribute == b"{attribute}"')["value"][0]
            return int(raw.decode('utf-8'))

        return ScanConfig(
            start_column=read_int('start_column'),
            stop_column=read_int('stop_column'),
            start_row=read_int('start_row'),
            stop_row=read_int('stop_row'),
            n_injections=read_int('n_injections'),
            vcal_low_start=read_int('VCAL_LOW_start'),
            vcal_low_stop=read_int('VCAL_LOW_stop'),
            vcal_low_step=read_int('VCAL_LOW_step'),
            vcal_high=read_int('VCAL_HIGH'),
        )


def compute_delta_v(config: ScanConfig) -> np.ndarray:
    return np.array(
        range(
            config.vcal_high - config.vcal_low_start,
            config.vcal_high - config.vcal_low_stop,
            -config.vcal_low_step,
        )
    )


def make_regions(config: ScanConfig) -> List[Region]:
    regions: List[Region] = []
    for frontend_start, frontend_stop, label in REGIONS:
        start_col = max(frontend_start, config.start_column)
        stop_col = min(frontend_stop, config.stop_column)
        start_row = max(0, config.start_row)
        stop_row = min(512, config.stop_row)
        if start_col < stop_col and start_row < stop_row:
            regions.append(
                Region(
                    start_col=start_col,
                    stop_col=stop_col,
                    start_row=start_row,
                    stop_row=stop_row,
                    label=label,
                    frontend_start_col=frontend_start,
                    frontend_stop_col=frontend_stop,
                )
            )
    return regions


def get_fit_start(args: argparse.Namespace) -> int:
    return args.fit_range


def is_region_active(region: Region, config: ScanConfig) -> bool:
    return region.start_col < region.stop_col and region.start_row < region.stop_row


def add_legend_if_needed(ax, fontsize: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=fontsize)


def read_histograms(file_path: str) -> Dict[str, np.ndarray]:
    with tb.open_file(file_path, 'r') as h5_file:
        return {
            'HistOcc': np.asarray(h5_file.root.HistOcc),
            'HistTot': np.asarray(h5_file.root.HistTot),
            'NoiseMap': np.asarray(h5_file.root.NoiseMap).T,
            'ThresholdMap': np.asarray(h5_file.root.ThresholdMap).T,
        }


def gauss(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - mean) ** 2) / (2.0 * sigma ** 2))


def func(charge: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
    charge = np.asarray(charge, dtype=float)
    inv_charge = np.divide(1.0, charge, out=np.zeros_like(charge, dtype=float), where=charge != 0)
    return (a * inv_charge + 1.0 / b) * (charge - d)


def inv_func(tot: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
    return (
        np.sqrt(b ** 2 * (a - tot) ** 2 + 2 * b * d * (a + tot) + d ** 2)
        - b * a
        + b * tot
        + d
    ) * 0.5


def derivate_func(charge: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
    return a * d / charge ** 2 + 1.0 / b


def chi_squared(
    fit_function,
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    params: Sequence[float],
    n_free_params: int,
    label: str,
) -> Tuple[float, int]:
    y_err = np.where(y_err == 0, 1e-9, y_err)
    residuals = (y - fit_function(x, *params)) / y_err
    chi2 = float(np.sum(residuals ** 2))
    ndof = len(y) - n_free_params
    print(f'{label} = {chi2:.3f} ({ndof} dof)')
    return chi2, ndof


def build_tot_map(hist_tot: np.ndarray, delta_v: np.ndarray, region: Region) -> np.ndarray:
    results = np.zeros((max(delta_v) + 1, 128), dtype=int)
    for step_index, charge in enumerate(delta_v):
        summed = np.sum(
            hist_tot[region.start_col:region.stop_col, region.start_row:region.stop_row, step_index, :],
            axis=(0, 1),
        )
        results[charge, :] = summed
    return results


def clean_data_by_charge(
    results: np.ndarray,
    nmin: float,
    max_tot_fit_bin: int,
    threshold_fraction: float,
    verbose: bool,
    debug_pdf: PdfPages | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_max = np.nanmax(results, axis=1, keepdims=True)
    threshold = row_max * threshold_fraction
    results_clean = np.where(results < threshold, np.nan, results)

    means: List[float] = []
    stds: List[float] = []
    rows: List[int] = []

    for tot_bin, row in enumerate(results_clean.T):
        fig, ax = plt.subplots()
        x = np.arange(len(row))
        ax.plot(x, row)
        ax.set_title(f'Charge distribution for TOT {tot_bin}')
        ax.set_xlabel('Charge DAC')
        ax.set_ylabel('Counts')
        ax.grid(True)

        try:
            p0 = [float(np.nanmax(row)), float(np.nanargmax(row)), 4.0]
        except ValueError:
            p0 = [0.0, 0.0, 0.0]

        n_entries = float(np.nansum(row))
        amplitude, mean, sigma_guess = p0

        if n_entries < nmin or tot_bin > max_tot_fit_bin:
            if verbose and n_entries != 0:
                print(f'skip ToT={tot_bin}, N={n_entries:.0f} < Nmin={nmin:.0f}')
            plt.close(fig)
            continue

        try:
            mask = ~np.isnan(row)
            x_clean = x[mask]
            row_clean = row[mask]
            if np.sum(row_clean) <= 0:
                plt.close(fig)
                continue

            popt, pcov = curve_fit(gauss, x_clean, row_clean, p0=p0)
            amplitude, mean, sigma_guess = popt
            perr = np.sqrt(np.diag(pcov))
            sigma = float(sigma_guess)

            x_fit = np.linspace(0, len(row) - 1, 500)
            ax.plot(x_fit, gauss(x_fit, *popt), 'r--', label=f'Gaussian fit, N={n_entries:.0f}')
            for idx, name in enumerate(['ampl', 'mean', 'std']):
                ax.annotate(
                    f'{name} = {popt[idx]:.2f} ± {perr[idx]:.2f}',
                    xy=(0.05, 0.95 - idx * 0.05),
                    xycoords='axes fraction',
                    fontsize=12,
                    color='k',
                    ha='left',
                    va='top',
                )
            add_legend_if_needed(ax, fontsize=10)
            fig.tight_layout()
            if debug_pdf is not None:
                debug_pdf.savefig(fig)
            plt.close(fig)

            if 0 < mean <= len(row):
                rows.append(tot_bin)
                means.append(float(mean))
                stds.append(sigma)
        except Exception as exc:
            if verbose:
                print(f'fit failed for ToT={tot_bin}: {exc}')
            if debug_pdf is not None:
                debug_pdf.savefig(fig)
            plt.close(fig)

    means_array = np.asarray(means)
    stds_array = np.asarray(stds)
    rows_array = np.asarray(rows)
    valid = ~np.isnan(means_array)

    if verbose:
        print('charge-mean calibration points')
        print('means')
        print([f'{value:.2f}' for value in means_array[valid]])
        print('stds')
        print([f'{value:.2f}' for value in stds_array[valid]])
        print('rows=TOT for which we have Qinj mean fitted')
        print([f'{value:.0f}' for value in rows_array[valid]])

    return means_array[valid], stds_array[valid], rows_array[valid], results_clean


def plot_scurves(pdf_file: pdf.PdfPages, hist_occ: np.ndarray, delta_v: np.ndarray, regions: List[Region], cmap: ListedColormap) -> None:
    print('\nplot s-curves')
    for region in regions:
        try:
            matrix = hist_occ[region.start_col:region.stop_col, region.start_row:region.stop_row]
            occupancy_values = list(range(int(np.max(matrix)) + 1))
            counts = np.zeros((matrix.shape[2], len(occupancy_values)))

            for step in range(matrix.shape[2]):
                for idx, occupancy in enumerate(occupancy_values):
                    counts[step, idx] = np.sum(matrix[:, :, step] == occupancy)

            x_values = delta_v[delta_v <= 700]
            counts = counts[:len(x_values)]

            fig, ax = plt.subplots()
            image = ax.pcolormesh(
                x_values,
                [value / 100.0 for value in occupancy_values],
                counts.T,
                cmap=cmap,
                norm=colors.LogNorm(vmin=1, vmax=np.max(counts)),
            )
            cbar = fig.colorbar(image, ax=ax, format=LogFormatterSciNotation(), ticks=ticker.LogLocator(base=10.0))
            cbar.set_label('# of pixels', fontsize=14)
            ax.set_xlabel('Injected Charge [DAC]', fontsize=16)
            ax.set_ylabel('Occupancy', fontsize=16)
            ax.set_title(f'S-Curve Plot {region.label}', fontsize=16)
            ax.tick_params(axis='both', which='both', labelsize=14)
            cbar.ax.tick_params(labelsize=14)
            plt.subplots_adjust(bottom=0.15)
            pdf_file.savefig(fig)
            plt.close(fig)
        except Exception as exc:
            print(f'Error while plotting s-curves for {region.label}: {exc}')


def fit_threshold_distribution(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hist, bins = np.histogram(data, bins=200)
    x_fit = (bins[1:-1] + bins[2:]) / 2.0
    y_fit = hist[1:]
    max_value = float(np.max(hist))
    nonzero = data[data != 0]
    center = float(np.mean(nonzero)) if nonzero.size else 0.0

    popt, pcov = curve_fit(gauss, x_fit, y_fit, p0=[max_value, center, 5.0])
    return x_fit, bins, popt, pcov


def plot_threshold_map(pdf_file: pdf.PdfPages, threshold_map: np.ndarray, config: ScanConfig) -> None:
    print('\nplot threshold map')
    threshold_view = threshold_map[config.start_row:config.stop_row, config.start_column:config.stop_column]
    fig, ax = plt.subplots()
    image = ax.imshow(
        threshold_view,
        vmin=26,
        vmax=29,
        extent=(config.start_column - 0.5, config.stop_column - 0.5, config.stop_row - 0.5, config.start_row - 0.5),
        aspect='auto',
        interpolation='nearest',
    )
    cbar = fig.colorbar(image, ax=ax, ticks=np.arange(26, 29, 1))
    cbar.ax.set_yticklabels(np.arange(26, 29, 1), fontsize=12)
    cbar.set_label('Threshold [DAC]', fontsize=14)
    ax.set_xlabel('Column', fontsize=14)
    ax.set_ylabel('Row', fontsize=14)
    ax.set_xticks(np.arange(config.start_column, config.stop_column, 1))
    row_span = config.stop_row - config.start_row
    if row_span <= 32:
        ax.set_yticks(np.arange(config.start_row, config.stop_row, 1))
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.tick_params(axis='both', which='both', labelsize=12)
    pdf_file.savefig(fig)
    plt.close(fig)


def plot_threshold_distributions(
    pdf_file: pdf.PdfPages,
    threshold_map: np.ndarray,
    regions: List[Region],
    verbose: bool,
) -> Dict[str, np.ndarray]:
    print('\nplot threshold')
    thresholds: Dict[str, np.ndarray] = {}

    for region in regions:
        fig = plt.figure()
        data = threshold_map[region.start_row:region.stop_row, region.start_col:region.stop_col]
        try:
            x_fit, bins, popt, _ = fit_threshold_distribution(data)
            x_plot = x_fit
            plt.plot(x_fit, gauss(x_fit, *popt), 'r-', label=f'{region.label} fit')
            plt.bar(x_plot, np.histogram(data, bins=200)[0][1:], width=bins[1] - bins[0], label=region.label)
            fig.text(
                0.15,
                0.55,
                f'µ = {popt[1]:.1f}DAC\n$\\sigma$ = {popt[2]:.1f}DAC',
                fontsize=16,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5),
            )
            if verbose:
                print(f'{region.label}: threshold={popt[1]:.1f} DAC, sigma={popt[2]:.1f} DAC')
            thresholds[region.label] = np.array([float(popt[1]), float(popt[2])])
        except Exception as exc:
            print(f'Could not fit threshold distribution for {region.label}: {exc}')
            thresholds[region.label] = np.array([0.0, 0.0])

        add_legend_if_needed(plt.gca(), fontsize=14)
        plt.xlabel('threshold [DAC]', fontsize=16)
        plt.ylabel('# pixel', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        pdf_file.savefig(fig)
        plt.close(fig)

    return thresholds


def plot_noise_map(pdf_file: pdf.PdfPages, noise_map: np.ndarray, config: ScanConfig) -> None:
    print('\nplot Noise map')
    noise_view = noise_map[config.start_row:config.stop_row, config.start_column:config.stop_column]
    fig, ax = plt.subplots()
    image = ax.imshow(
        noise_view,
        vmin=2,
        vmax=4,
        extent=(config.start_column - 0.5, config.stop_column - 0.5, config.stop_row - 0.5, config.start_row - 0.5),
        aspect='auto',
        interpolation='nearest',
    )
    cbar = fig.colorbar(image, ax=ax, ticks=np.arange(2, 4, 1))
    cbar.ax.set_yticklabels(np.arange(2, 4, 1), fontsize=12)
    cbar.set_label('Noise [DAC]', fontsize=14)
    ax.set_xlabel('Column', fontsize=14)
    ax.set_ylabel('Row', fontsize=14)
    ax.set_xticks(np.arange(config.start_column, config.stop_column, 1))
    row_span = config.stop_row - config.start_row
    if row_span <= 32:
        ax.set_yticks(np.arange(config.start_row, config.stop_row, 1))
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    pdf_file.savefig(fig)
    plt.close(fig)


def plot_noise_distributions(pdf_file: pdf.PdfPages, noise_map: np.ndarray, regions: List[Region]) -> None:
    print('\nplot Noise')
    for region in regions:
        fig = plt.figure()
        data = noise_map[region.start_row:region.stop_row, region.start_col:region.stop_col]
        data = data[np.isfinite(data)]
        data = data[data > 0]
        if data.size == 0:
            plt.close(fig)
            continue
        hist_noise, bins_noise = np.histogram(data, bins=50)
        x_fit_noise = (bins_noise[1:-1] + bins_noise[2:]) / 2.0
        x_plot_noise = (bins_noise[1:-1] + bins_noise[2:]) / 2.0
        try:
            peak_index = int(np.argmax(hist_noise[1:])) if len(hist_noise) > 1 else int(np.argmax(hist_noise))
            center_guess = x_plot_noise[peak_index] if len(x_plot_noise) else float(np.mean(data))
            sigma_guess = max(float(np.std(data)), 0.2)
            p0_noise = [float(np.max(hist_noise)), center_guess, sigma_guess]
            lower_bounds = [0.0, float(np.min(data)), 0.01]
            upper_bounds = [np.inf, float(np.max(data)), max(float(np.max(data) - np.min(data)), 0.05)]
            popt_noise, _ = curve_fit(
                gauss,
                x_fit_noise,
                hist_noise[1:],
                p0=p0_noise,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000,
            )
            plt.plot(x_fit_noise, gauss(x_fit_noise, *popt_noise), 'r-', label=f'{region.label} fit')
            plt.bar(x_plot_noise, hist_noise[1:], width=bins_noise[1] - bins_noise[0], label=region.label)
            fig.text(
                0.15,
                0.55,
                f'µ = {popt_noise[1]:.2f}DAC\n$\\sigma$ = {popt_noise[2]:.2f}DAC',
                fontsize=14,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5),
            )
        except Exception as exc:
            print(f'Could not fit noise distribution for {region.label}: {exc}')

        add_legend_if_needed(plt.gca(), fontsize=16)
        plt.xlabel('Noises [DAC]', fontsize=16)
        plt.ylabel('# pixel', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        pdf_file.savefig(fig)
        plt.close(fig)


def run_region_fit(
    pdf_file: pdf.PdfPages,
    debug_pdf: PdfPages | None,
    region: Region,
    region_index: int,
    tot_map: np.ndarray,
    threshold_values: np.ndarray,
    args: argparse.Namespace,
    cmap: ListedColormap,
    nmin: float,
) -> RegionCalibration:
    fit_start = get_fit_start(args)
    means, stds, rows, results_clean = clean_data_by_charge(
        tot_map,
        nmin=nmin,
        max_tot_fit_bin=args.max_tot_fit_bin,
        threshold_fraction=args.threshold_fraction,
        verbose=args.verbose,
        debug_pdf=debug_pdf,
    )
    if len(rows) == 0:
        raise RuntimeError(f'No valid calibration points for {region.label}')

    tot_mask = rows >= fit_start
    charge_mean = means[tot_mask]
    charge_err = stds[tot_mask]
    tot_int = rows[tot_mask]
    tot_err = np.full_like(tot_int, TOT_BIN_ERROR, dtype=float)
    if len(tot_int) == 0:
        raise RuntimeError(f'No calibration points left after fit range cut for {region.label}')

    if args.verbose:
        print('tot_int')
        print([f'{value:.2f}' for value in tot_int])
        print('tot_err')
        print([f'{value:.2f}' for value in tot_err])
        print('charge_mean')
        print([f'{value:.2f}' for value in charge_mean])
        print('charge_err')
        print([f'{value:.2f}' for value in charge_err])

    fig = plt.figure(figsize=(8, 6))
    x = np.arange(tot_map.shape[0], dtype=float)
    plot_start = max(args.plot_range, 1)
    x_plot = np.linspace(x[plot_start], 2500, 1000)
    y = np.arange(tot_map.shape[1])
    mesh = plt.pcolormesh(x, y, results_clean.T[:, -len(x):], cmap=cmap, norm=colors.LogNorm())
    colorbar = plt.colorbar(mesh)
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label('# of pixel', fontsize=14)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=45, fontsize=12)
    plt.errorbar(charge_mean, tot_int, yerr=tot_err, fmt='ro', label='mean charge', markersize=3)

    threshold_value = float(threshold_values[0])
    threshold_sigma = float(threshold_values[1])
    print(f'{region.label}: threshold={threshold_value:.1f} DAC, sigma={threshold_sigma:.1f} DAC, fit points={len(tot_int)}')
    calibration_curve_y_limit = max(2.5, 2.5 * float(np.max(tot_int)))
    tot_map_y_limit = max(5.0, float(np.ceil(2.0 * np.max(tot_int))))

    if args.fit_mode == 'fit2par':
        p0 = (7.0, 70.0)
        bounds = ([0.0, 0.0], [np.inf, np.inf])
        popt_fit, pcov_fit = curve_fit(
            lambda charge, a, b: func(charge, a, b, threshold_value),
            charge_mean,
            tot_int,
            sigma=tot_err,
            p0=p0,
            bounds=bounds,
        )
        fit_params = np.array([popt_fit[0], popt_fit[1], threshold_value])
        free_param_names = ['a', 'b']
        label_text = '$f_{2par}(x)=(a/x +1/b)*(x-THR)$'
        chi2_val, ndof = chi_squared(func, charge_mean, tot_int, tot_err, fit_params, 2, label=f'Chi² - {region.label}')
    else:
        p0 = (4.45, 100.0, threshold_value)
        bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])
        fit_params, pcov_fit = curve_fit(
            func,
            charge_mean,
            tot_int,
            sigma=tot_err,
            p0=p0,
            bounds=bounds,
        )
        free_param_names = ['a', 'b', 'd']
        label_text = '$f_{3par}(x)=(a/x +1/b)*(x-d)$'
        chi2_val, ndof = chi_squared(func, charge_mean, tot_int, tot_err, fit_params, 3, label=f'Chi² - {region.label}')

    chi2_ndof = chi2_val / ndof if ndof > 0 else 0.0
    plt.plot(x_plot, func(x_plot, *fit_params), color='k', label='fit')
    terminal_params: List[str] = []
    for idx, name in enumerate(free_param_names):
        err = np.sqrt(pcov_fit[idx, idx]) if idx < pcov_fit.shape[0] else 0.0
        terminal_params.append(f'{name}={fit_params[idx]:.2f}±{err:.2f}')
    if args.fit_mode == 'fit2par':
        terminal_params.append(f'THR={threshold_value:.2f}')
    else:
        slope = 1 / fit_params[1]
        cost = fit_params[0] - fit_params[2] / fit_params[1]
        c_par = fit_params[0] * fit_params[2]
        terminal_params.extend([f'slope={slope:.2f}', f'cost={cost:.2f}', f'c={c_par:.2f}'])
        if args.verbose:
            pass

    summary_lines = [
        f'Frontend: {region.label}',
        f'Fit mode: {args.fit_mode}',
        f'Formula: {label_text}',
        f'Threshold = {threshold_value:.2f} DAC',
        f'Threshold sigma = {threshold_sigma:.2f} DAC',
        f'Fit points = {len(tot_int)}',
        f'Chi2/ndof = {chi2_val:.3f}/{ndof} = {chi2_ndof:.3f}',
    ]
    summary_lines.extend(terminal_params)
    if args.fit_mode == 'fit3par':
        summary_lines.extend([
            '',
            'Equivalent 4-par terms assuming t = 0:',
            'slope = a_4par = 1 / b',
            'cost = b_4par = a - d / b',
            'c = c_4par = a * d',
        ])

    plt.xlabel('Injected Charge [DAC]', fontsize=14)
    plt.ylabel('ToT  [25ns]', fontsize=14)
    plt.title(region.label)
    plt.ylim(0, tot_map_y_limit)
    plt.xlim(0, 250)
    plt.gca().set_aspect('auto', adjustable='box')
    plt.tick_params(axis='both', which='both', labelsize=12)
    add_legend_if_needed(plt.gca(), fontsize=14)
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.12, top=0.92)
    pdf_file.savefig(fig)
    plt.close(fig)

    fig2 = plt.figure(figsize=(8, 6))
    ax1, ax2 = fig2.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
    ax1.errorbar(charge_mean, tot_int, yerr=np.abs(tot_err), fmt='o', label='Fit Data', color='blue', markersize=6)
    ax1.plot(x_plot, func(x_plot, *fit_params), 'k--', label='Fit: $f(x)$')
    tot_test = np.arange(0, 21)
    ax1.plot(inv_func(tot_test, *fit_params), tot_test, 'go', label='inverted func: $inv_f(int ToT)$')
    ax1.set_ylabel('ToT Mean [25ns]', fontsize=14)
    ax1.set_title(f'Calibration Curve Fit - {region.label}', fontsize=16)
    ax1.set_ylim(0, calibration_curve_y_limit)
    ax1.legend(fontsize=12)
    ax1.grid(True)

    residuals = tot_int - func(charge_mean, *fit_params)
    ax2.errorbar(charge_mean, residuals, yerr=np.abs(tot_err), color='blue', fmt='o')
    ax2.plot(x_plot, np.zeros_like(x_plot), '--', color='black')
    ax2.set_xlim(0, 250)
    ax2.set_ylabel('Residuals ToT [25ns]', fontsize=14)
    ax2.set_xlabel('Injected Charge [DAC]', fontsize=14)
    ax2.grid(color='lightgray', ls='dashed')
    fig2.subplots_adjust(hspace=0.08)
    pdf_file.savefig(fig2)
    plt.close(fig2)

    fig_summary = plt.figure(figsize=(8, 6))
    ax_summary = fig_summary.add_subplot(111)
    ax_summary.axis('off')
    ax_summary.text(
        0.06,
        0.94,
        f'Calibration Summary - {region.label}',
        fontsize=16,
        va='top',
        ha='left',
        fontweight='bold',
    )
    ax_summary.text(
        0.06,
        0.86,
        '\n'.join(summary_lines),
        fontsize=11,
        va='top',
        ha='left',
        linespacing=1.45,
        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9),
    )
    pdf_file.savefig(fig_summary)
    plt.close(fig_summary)

    fig3 = plt.figure(figsize=(8, 6))
    mesh = plt.pcolormesh(x, y, tot_map.T[:, -len(x):], cmap=cmap, norm=colors.LogNorm())
    colorbar = plt.colorbar(mesh)
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label('# of pixel', fontsize=14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=45, fontsize=12)
    plt.errorbar(charge_mean, tot_int, yerr=tot_err, fmt='ro', label='mean charge', markersize=3)
    plt.plot(x_plot, func(x_plot, *fit_params), color='k', label='fit')
    plt.text(10, 40, label_text, color='k', fontsize=14)
    plt.xlabel('Injected Charge [DAC]', fontsize=14)
    plt.ylabel('ToT  [25ns]', fontsize=14)
    plt.title(f'Raw data {region.label}')
    plt.ylim(0, tot_map_y_limit)
    plt.xlim(0, 250)
    plt.gca().set_aspect('auto', adjustable='box')
    plt.tick_params(axis='both', which='both', labelsize=12)
    add_legend_if_needed(plt.gca(), fontsize=14)
    fig3.subplots_adjust(left=0.11, right=0.96, bottom=0.12, top=0.92)
    plt.grid(True)
    pdf_file.savefig(fig3)
    plt.close(fig3)

    print(f'{region.label}: fit ok, chi2/ndof={chi2_ndof:.3f}, ' + ', '.join(terminal_params))

    return RegionCalibration(np.asarray(fit_params), chi2_ndof)


def build_output_array(region_calibrations: List[RegionCalibration]) -> np.ndarray:
    inj_tot_cal = np.zeros((512, 512, 4))
    for (start_col, stop_col, _), calibration in zip(REGIONS, region_calibrations):
        inj_tot_cal[start_col:stop_col, :, 0:3] = calibration.fit_params
        inj_tot_cal[start_col:stop_col, :, 3] = calibration.chi2_ndof
    return inj_tot_cal


def write_output_h5(input_file: str, fit_mode: str, output_array: np.ndarray) -> str:
    folder_path = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(folder_path, f'{base_name}_tot_calibration_{fit_mode}_charge_mean.h5')
    shutil.copy(input_file, output_file)

    with tb.open_file(output_file, 'r+') as out_file:
        if hasattr(out_file.root, 'InjTotCalibration'):
            out_file.remove_node(out_file.root, 'InjTotCalibration')
        out_file.create_carray(
            out_file.root,
            name='InjTotCalibration',
            title='Injection Tot Calibration Fit',
            obj=output_array,
            filters=tb.Filters(complib='blosc', complevel=5),
        )
        out_file.root.InjTotCalibration.attrs.columns = ['a', 'b', 'd', 'chi2_ndof']

    return output_file


def main() -> None:
    args = build_parser().parse_args()
    file_path = resolve_input_file(args.input_file)
    folder_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    print(f'[INFO] File selezionato: {file_name}')
    print(f'[INFO] Cartella: {folder_path}')

    config = read_scan_config(file_path)
    delta_v = compute_delta_v(config)
    all_regions = make_regions(config)
    active_regions = [region for region in all_regions if is_region_active(region, config)]
    active_region_labels = [region.label for region in active_regions]
    histograms = read_histograms(file_path)
    cmap = build_colormap()

    print('Configuration:')
    print(
        f'cols={config.start_column}:{config.stop_column}, rows={config.start_row}:{config.stop_row}, '
        f'n_inj={config.n_injections}, VH={config.vcal_high}, VL_start={config.vcal_low_start}, '
        f'VL_stop={config.vcal_low_stop}, VL_step={config.vcal_low_step}, nsteps={len(delta_v)}'
    )
    nmin = (config.stop_column - config.start_column) * (config.stop_row - config.start_row) * config.n_injections / 5.0
    print(f'Active frontends: {", ".join(active_region_labels) if active_region_labels else "none"}')
    print(f'Nmin={nmin:.1f}')

    pdf_path = os.path.join(folder_path, f'{base_name}_tot_calibration_{args.fit_mode}_charge_mean.pdf')
    charge_debug_pdf_path = os.path.join(folder_path, f'{base_name}_charge_DAC_tot_distribution_ALL.pdf')
    region_calibrations: Dict[str, RegionCalibration] = {}

    with pdf.PdfPages(pdf_path) as pdf_file, PdfPages(charge_debug_pdf_path) if args.save_charge_debug_pdf else nullcontext() as debug_pdf:
        plot_scurves(pdf_file, histograms['HistOcc'], delta_v, active_regions, cmap)
        plot_threshold_map(pdf_file, histograms['ThresholdMap'], config)
        thresholds = plot_threshold_distributions(pdf_file, histograms['ThresholdMap'], active_regions, args.verbose)
        plot_noise_map(pdf_file, histograms['NoiseMap'], config)
        plot_noise_distributions(pdf_file, histograms['NoiseMap'], active_regions)

        print('\nplot ToT')
        for region_index, region in enumerate(all_regions):
            if region.label not in active_region_labels:
                continue
            tot_map = build_tot_map(histograms['HistTot'], delta_v, region)
            try:
                calibration = run_region_fit(
                    pdf_file=pdf_file,
                    debug_pdf=debug_pdf if args.save_charge_debug_pdf else None,
                    region=region,
                    region_index=region_index,
                    tot_map=tot_map,
                    threshold_values=thresholds[region.label],
                    args=args,
                    cmap=cmap,
                    nmin=nmin,
                )
            except Exception as exc:
                print(f'{region.label}: skipped ({exc})')
                threshold_value = float(thresholds[region.label][0])
                calibration = RegionCalibration(np.array([0.0, 0.0, threshold_value]), 0.0)
            region_calibrations[region.label] = calibration

    ordered_calibrations = [
        region_calibrations.get(region_label, RegionCalibration(np.zeros(3), 0.0))
        for _, _, region_label in REGIONS
    ]
    output_array = build_output_array(ordered_calibrations)
    output_h5 = write_output_h5(file_path, args.fit_mode, output_array)
    print(f'[INFO] Created PDF: {pdf_path}')
    if args.save_charge_debug_pdf:
        print(f'[INFO] Created charge debug PDF: {charge_debug_pdf_path}')
    print(f'[INFO] Created H5: {output_h5}')
    if active_region_labels:
        threshold_summary = {label: [f'{values[0]:.1f}', f'{values[1]:.1f}'] for label, values in thresholds.items()}
        print('Thresholds (active frontends):', threshold_summary)


if __name__ == '__main__':
    main()
