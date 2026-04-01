import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.backends.backend_pdf as pdf_backend
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tables as tb
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit

DEFAULT_PATH = '/home/labb2/tj-monopix2-daq-development/tjmonopix2/scans/output_data/module_0/chip_0/'
REGIONS = (
    (0, 224, 'NF'),
    (224, 448, 'NF CASC'),
    (448, 480, 'HV CASC'),
    (480, 512, 'HV'),
)
SCURVE_MAX_CHARGE_DAC = 700
MAX_TOT_BIN_FOR_FIT = 120
THRESHOLD_FILTER_FRACTION = 0.05
TOT_BIN_ERROR = 1.0 / np.sqrt(12.0)


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


@dataclass
class RegionSummary:
    threshold_mean: float
    threshold_sigma: float
    fit_params: np.ndarray
    chi2_ndof: float


class LogFormatterSciNotation(ticker.LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        return r'$10^{{{}}}$'.format(int(np.log10(x)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Analyse ToT calibration from interpreted threshold scan files.'
    )
    parser.add_argument(
        '-f', '--input_file',
        type=str,
        default=None,
        help='Full path to the HDF5 file or a file name inside the default scan directory.'
    )
    parser.add_argument('-fit_HV', '--fit_range_HV', type=int, default=0, help='Unused legacy option.')
    parser.add_argument('-fit_HVCASC', '--fit_range_HVCASC', type=int, default=0, help='Unused legacy option.')
    parser.add_argument('-fit_NF', '--fit_range_NF', type=int, default=0, help='Unused legacy option.')
    parser.add_argument('-fit_NFCASC', '--fit_range_NFCASC', type=int, default=0, help='Unused legacy option.')
    parser.add_argument('-plot', '--plot_range', type=int, default=0, help='First x-bin shown in calibration plots.')
    parser.add_argument(
        '-center', '--center', type=int, default=0,
        help='Optional manual threshold center. Kept for compatibility with the original scripts.'
    )
    return parser


def build_colormap() -> ListedColormap:
    viridis = colormaps.get_cmap('viridis').resampled(256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])
    return ListedColormap(newcolors)


def find_latest_file(directory: str, partial_name: str) -> Optional[str]:
    files = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if partial_name in name and os.path.isfile(os.path.join(directory, name))
    ]
    if not files:
        return None
    return max(files, key=os.path.getctime)


def resolve_input_file(user_input: Optional[str]) -> str:
    if user_input is None:
        latest = find_latest_file(DEFAULT_PATH, '_threshold_scan_interpreted.h5')
        if latest is None:
            print(f'[INFO] No interpreted file found in {DEFAULT_PATH}.')
            sys.exit(0)
        print(f'[INFO] No file specified, using latest file: {latest}')
        return latest

    if os.sep not in user_input:
        candidate = os.path.join(DEFAULT_PATH, user_input)
    else:
        candidate = user_input

    if not os.path.exists(candidate):
        print(f'[ERROR] File not found: {candidate}')
        sys.exit(1)

    return candidate


def read_scan_config(h5_file: tb.File) -> ScanConfig:
    table = h5_file.root.configuration_in.scan.scan_config

    def read_int(attribute: str) -> int:
        raw_value = table.read_where(f'attribute == b"{attribute}"')["value"][0]
        return int(raw_value.decode('utf-8'))

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


def make_regions(config: ScanConfig) -> List[Region]:
    return [
        Region(start_col, stop_col, config.start_row, config.stop_row, label)
        for start_col, stop_col, label in REGIONS
    ]


def compute_delta_v(config: ScanConfig) -> np.ndarray:
    return np.array(
        range(
            config.vcal_high - config.vcal_low_start,
            config.vcal_high - config.vcal_low_stop,
            -config.vcal_low_step,
        )
    )


def read_hist_arrays(file_path: str) -> Dict[str, np.ndarray]:
    with tb.open_file(file_path, 'r') as h5_file:
        return {
            'HistOcc': np.asarray(h5_file.root.HistOcc),
            'HistTot': np.asarray(h5_file.root.HistTot),
            'NoiseMap': np.asarray(h5_file.root.NoiseMap).T,
            'ThresholdMap': np.asarray(h5_file.root.ThresholdMap).T,
        }


def gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - mean) ** 2) / (2.0 * sigma ** 2))


def calibration_func(charge: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
    return (a / charge + 1.0 / b) * (charge - d)


def inv_calibration_func(tot: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
    return (
        np.sqrt(b ** 2 * (a - tot) ** 2 + 2 * b * d * (a + tot) + d ** 2)
        - b * a
        + b * tot
        + d
    ) * 0.5


def chi_squared(
    func: Callable[..., np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    params: Sequence[float],
    n_free_params: int,
    label: str,
) -> Tuple[float, int]:
    dy_safe = np.where(dy == 0, 1e-9, dy)
    residuals = (y - func(x, *params)) / dy_safe
    chi2 = float(np.sum(residuals ** 2))
    ndof = len(y) - n_free_params
    print(f'{label} = {chi2:.3f} ({ndof} dof)')
    return chi2, ndof


def fit_threshold_distribution(
    data: np.ndarray,
    center_arg: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    hist, bins = np.histogram(data, bins=200)
    x_fit = (bins[1:-1] + bins[2:]) / 2.0
    y_fit = hist[1:]
    max_value = np.max(hist)

    if center_arg == 0:
        nonzero_data = data[data != 0]
        center = float(np.mean(nonzero_data)) if nonzero_data.size > 0 else 0.0
    else:
        center = center_arg / 10.1

    if max_value == 0:
        return None

    p0 = [max_value, center, 5.0]
    popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0)
    return x_fit, y_fit, popt, pcov, bins


def build_tot_map(hist_tot: np.ndarray, delta_v: np.ndarray, region: Region) -> np.ndarray:
    results = np.zeros((max(delta_v) + 1, 128), dtype=int)
    for step_index, charge in enumerate(delta_v):
        summed = np.sum(
            hist_tot[region.start_col:region.stop_col, region.start_row:region.stop_row, step_index, :],
            axis=(0, 1),
        )
        results[charge, :] = summed
    return results


def clean_tot_results_by_charge(results: np.ndarray, nmin: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_max = np.nanmax(results, axis=1, keepdims=True)
    threshold = row_max * THRESHOLD_FILTER_FRACTION
    results_clean = np.where(results < threshold, np.nan, results)

    means: List[float] = []
    stds: List[float] = []
    rows: List[int] = []

    with PdfPages('charge_DAC_tot_distribution_ALL.pdf') as pdf_file:
        for tot_bin, row in enumerate(results_clean.T):
            x = np.arange(len(row))
            valid_mask = ~np.isnan(row)
            x_clean = x[valid_mask]
            row_clean = row[valid_mask]
            n_entries = np.nansum(row)

            if n_entries < nmin or tot_bin > MAX_TOT_BIN_FOR_FIT or np.sum(row_clean) <= 0:
                if n_entries != 0:
                    print(f'skip ToT={tot_bin}, N={n_entries} < Nmin={nmin}')
                continue

            fig, ax = plt.subplots()
            ax.plot(np.arange(len(row)), row)
            ax.set_title(f'Charge distribution for TOT {tot_bin}')
            ax.set_xlabel('Charge DAC')
            ax.set_ylabel('Counts')
            ax.grid(True)

            p0 = [float(np.nanmax(row)), float(np.nanargmax(row)), 4.0]
            try:
                popt, pcov = curve_fit(gaussian, x_clean, row_clean, p0=p0)
            except Exception:
                pdf_file.savefig(fig)
                plt.close(fig)
                print('fit failed')
                continue

            amplitude, mean, sigma = popt
            x_fit = np.linspace(0, len(row) - 1, 500)
            ax.plot(x_fit, gaussian(x_fit, *popt), 'r--', label=f'Gaussian fit, N={n_entries:.0f}')
            for idx, name in enumerate(['ampl', 'mean', 'std']):
                ax.annotate(
                    f'{name} = {popt[idx]:.2f} ± {np.sqrt(pcov[idx, idx]):.2f}',
                    xy=(0.05, 0.95 - idx * 0.05),
                    xycoords='axes fraction',
                    fontsize=12,
                    color='k',
                    ha='left',
                    va='top',
                )
            ax.legend()
            fig.tight_layout()
            pdf_file.savefig(fig)
            plt.close(fig)

            if 0 < mean <= len(row):
                rows.append(tot_bin)
                means.append(float(mean))
                stds.append(float(sigma))

    means_array = np.asarray(means)
    stds_array = np.asarray(stds)
    rows_array = np.asarray(rows)
    valid = ~np.isnan(means_array)
    return means_array[valid], stds_array[valid], rows_array[valid], results_clean


def plot_scurves(pdf_file: pdf_backend.PdfPages, hist_occ: np.ndarray, delta_v: np.ndarray, regions: List[Region], cmap: ListedColormap) -> None:
    print('\nplot s-curves')
    for region in regions:
        try:
            matrix = hist_occ[region.start_col:region.stop_col, region.start_row:region.stop_row]
            occupancy_values = list(range(int(np.max(matrix)) + 1))
            counts = np.zeros((matrix.shape[2], len(occupancy_values)))

            for step in range(matrix.shape[2]):
                for idx, occupancy in enumerate(occupancy_values):
                    counts[step, idx] = np.sum(matrix[:, :, step] == occupancy)

            x_values = delta_v[delta_v <= SCURVE_MAX_CHARGE_DAC]
            counts = counts[:len(x_values)]

            fig, ax = plt.subplots()
            image = ax.pcolormesh(
                x_values,
                [value / 100.0 for value in occupancy_values],
                counts.T,
                cmap=cmap,
                norm=mcolors.LogNorm(vmin=1, vmax=np.max(counts)),
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
            print(f'Error while plotting S-curves for {region.label}: {exc}')


def plot_threshold_map(pdf_file: pdf_backend.PdfPages, threshold_map: np.ndarray) -> None:
    print('\nplot threshold map')
    fig, ax = plt.subplots()
    image = ax.imshow(threshold_map, vmin=26, vmax=29)
    cbar = fig.colorbar(image, ax=ax, ticks=np.arange(26, 29, 1))
    cbar.ax.set_yticklabels(np.arange(26, 29, 1), fontsize=12)
    cbar.set_label('Threshold [DAC]', fontsize=14)
    ax.set_xlabel('Column', fontsize=14)
    ax.set_ylabel('Row', fontsize=14)
    ax.tick_params(axis='both', which='both', labelsize=12)
    pdf_file.savefig(fig)
    plt.close(fig)


def plot_threshold_distributions(
    pdf_file: pdf_backend.PdfPages,
    threshold_map: np.ndarray,
    regions: List[Region],
    center_arg: int,
) -> np.ndarray:
    print('\nplot threshold')
    thresholds: List[List[float]] = []

    for region in regions:
        data = threshold_map[region.start_row:region.stop_row, region.start_col:region.stop_col]
        fit_result = fit_threshold_distribution(data, center_arg)
        if fit_result is None:
            thresholds.append([0.0, 0.0])
            continue

        x_fit, y_fit, popt, pcov, bins = fit_result
        fig = plt.figure()
        plt.bar(x_fit, y_fit, width=bins[1] - bins[0], label=region.label)
        plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', label=f'{region.label} fit')
        fig.text(
            0.15,
            0.55,
            f'µ = {popt[1]:.1f}DAC\n$\\sigma$ = {popt[2]:.1f}DAC',
            fontsize=16,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5),
        )
        plt.legend(fontsize=14)
        plt.xlabel('threshold [DAC]', fontsize=16)
        plt.ylabel('# pixel', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        pdf_file.savefig(fig)
        plt.close(fig)
        thresholds.append([float(popt[1]), float(popt[2])])

    threshold_array = np.asarray(thresholds)
    print('thresholds all regions', [[f'{thr:.1f}', f'{std:.1f}'] for thr, std in threshold_array])
    return threshold_array


def plot_noise_map(pdf_file: pdf_backend.PdfPages, noise_map: np.ndarray) -> None:
    print('\nplot noise map')
    fig, ax = plt.subplots()
    image = ax.imshow(noise_map, vmin=2, vmax=4)
    cbar = fig.colorbar(image, ax=ax, ticks=np.arange(2, 4, 1))
    cbar.ax.set_yticklabels(np.arange(2, 4, 1), fontsize=12)
    cbar.set_label('Noise [DAC]', fontsize=14)
    ax.set_xlabel('Column', fontsize=14)
    ax.set_ylabel('Row', fontsize=14)
    pdf_file.savefig(fig)
    plt.close(fig)


def plot_noise_distributions(pdf_file: pdf_backend.PdfPages, noise_map: np.ndarray, regions: List[Region]) -> None:
    print('\nplot noise')
    for region in regions:
        data = noise_map[region.start_row:region.stop_row, region.start_col:region.stop_col]
        hist, bins = np.histogram(data, bins=50)
        x_fit = (bins[1:-1] + bins[2:]) / 2.0
        y_fit = hist[1:]

        fig = plt.figure()
        try:
            p0 = [float(np.max(hist)), float(np.mean(data)), 10.0]
            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
            if popt[1] < 1000:
                plt.bar(x_fit, y_fit, width=bins[1] - bins[0], label=region.label)
                plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', label=f'{region.label} fit')
                fig.text(
                    0.15,
                    0.55,
                    f'µ = {popt[1]:.2f}DAC\n$\\sigma$ = {popt[2]:.2f}DAC',
                    fontsize=14,
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.5),
                )
        except Exception:
            pass

        plt.legend(fontsize=16)
        plt.xlabel('Noise [DAC]', fontsize=16)
        plt.ylabel('# pixel', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        pdf_file.savefig(fig)
        plt.close(fig)


def plot_calibration_summary(
    pdf_file: pdf_backend.PdfPages,
    section_name: str,
    x_plot: np.ndarray,
    tot_map: np.ndarray,
    results_clean: np.ndarray,
    charge_mean: np.ndarray,
    tot_int: np.ndarray,
    fit_curve_params: Sequence[float],
    displayed_params: Sequence[float],
    free_param_names: Sequence[str],
    free_cov: np.ndarray,
    chi2_val: float,
    ndof: int,
    model_label: str,
    cmap: ListedColormap,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    x = np.arange(tot_map.shape[0], dtype=float)
    y = np.arange(tot_map.shape[1])
    mesh = plt.pcolormesh(x, y, results_clean.T[:, -len(x):], cmap=cmap, norm=mcolors.LogNorm())
    colorbar = plt.colorbar(mesh)
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label('# of pixel', fontsize=14)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=45, fontsize=12)
    plt.errorbar(charge_mean, tot_int, yerr=np.full_like(tot_int, TOT_BIN_ERROR, dtype=float), fmt='ro', label='mean charge', markersize=3)
    plt.plot(x_plot, calibration_func(x_plot, *fit_curve_params), color='k', label='fit')
    plt.text(10, 40, model_label, color='k', fontsize=14)

    for idx, name in enumerate(free_param_names):
        plt.text(10, 35 - idx * 5, f'{name}={displayed_params[idx]:.2f} ± {np.sqrt(free_cov[idx, idx]):.2f}', color='k', fontsize=14)
    if len(displayed_params) > len(free_param_names):
        plt.text(10, 35 - len(free_param_names) * 5, f'd={displayed_params[-1]:.2f}', color='k', fontsize=14)

    plt.text(
        0.05,
        0.90,
        f'$\\chi^2$/ndof = {chi2_val:.3f}/{ndof} = {chi2_val / ndof:.3f}',
        transform=plt.gca().transAxes,
        fontsize=12,
        color='black',
    )
    plt.xlabel('Injected Charge [DAC]', fontsize=14)
    plt.ylabel('ToT [25ns]', fontsize=14)
    plt.title(section_name)
    plt.ylim(0, 50)
    plt.xlim(0, 250)
    plt.gca().set_aspect('auto', adjustable='box')
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.legend(fontsize=14)
    plt.tight_layout()
    pdf_file.savefig(fig)
    plt.close(fig)

    fig2 = plt.figure(figsize=(8, 6))
    ax1, ax2 = fig2.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
    ax1.errorbar(charge_mean, tot_int, yerr=np.full_like(tot_int, TOT_BIN_ERROR, dtype=float), fmt='o', label='Fit Data', color='blue', markersize=6)
    ax1.plot(x_plot, calibration_func(x_plot, *fit_curve_params), 'k--', label='Fit: $f(x)$')
    tot_test = np.arange(0, 21)
    ax1.plot(inv_calibration_func(tot_test, *fit_curve_params), tot_test, 'go', label='inverted func: $inv_f(int ToT)$')
    ax1.set_ylabel('ToT Mean [25ns]', fontsize=14)
    ax1.set_title(f'Calibration Curve Fit - {section_name}', fontsize=16)
    ax1.set_ylim(0, 10)
    ax1.legend(fontsize=12)
    ax1.grid(True)

    residuals = tot_int - calibration_func(charge_mean, *fit_curve_params)
    ax2.errorbar(charge_mean, residuals, yerr=np.full_like(tot_int, TOT_BIN_ERROR, dtype=float), color='blue', fmt='o')
    ax2.plot(x_plot, np.zeros_like(x_plot), '--', color='black')
    ax2.set_xlim(0, 250)
    ax2.set_xlabel('Injected Charge [DAC]', fontsize=14)
    ax2.set_ylabel('Residuals ToT [25ns]', fontsize=14)
    ax2.grid(color='lightgray', ls='dashed')
    plt.tight_layout()
    pdf_file.savefig(fig2)
    plt.close(fig2)


def build_inj_tot_calibration(region_summaries: List[RegionSummary]) -> np.ndarray:
    inj_tot_cal = np.zeros((512, 512, 4))
    for region, summary in zip(REGIONS, region_summaries):
        start_col, stop_col, _ = region
        inj_tot_cal[start_col:stop_col, :, 0:3] = summary.fit_params
        inj_tot_cal[start_col:stop_col, :, 3] = summary.chi2_ndof

    # Keep the same stored column names as the original files.
    # In the 2-par case the third value is the fixed threshold used during the fit.
    return inj_tot_cal


def write_output_h5(input_file: str, base_name: str, fit_mode: str, inj_tot_cal: np.ndarray) -> str:
    folder_path = os.path.dirname(input_file)
    output_file = os.path.join(folder_path, f'{base_name}_tot_calibration_{fit_mode}_charge_mean.h5')
    shutil.copy(input_file, output_file)

    with tb.open_file(output_file, 'r+') as out_file:
        if hasattr(out_file.root, 'InjTotCalibration'):
            out_file.remove_node(out_file.root, 'InjTotCalibration')
        out_file.create_carray(
            out_file.root,
            name='InjTotCalibration',
            title='Injection Tot Calibration Fit',
            obj=inj_tot_cal,
            filters=tb.Filters(complib='blosc', complevel=5),
        )
        out_file.root.InjTotCalibration.attrs.columns = ['a', 'b', 'd', 'chi2_ndof']

    return output_file


def fit_region_calibration(
    section_name: str,
    tot_map: np.ndarray,
    thresholds: np.ndarray,
    region_index: int,
    args: argparse.Namespace,
    fit_mode: str,
    pdf_file: pdf_backend.PdfPages,
    cmap: ListedColormap,
    nmin: float,
) -> RegionSummary:
    charge_mean, charge_err, tot_int, results_clean = clean_tot_results_by_charge(tot_map, nmin)
    if len(charge_mean) == 0 or len(tot_int) == 0:
        print(f'[WARNING] No calibration points for {section_name}.')
        threshold_value = float(thresholds[region_index, 0])
        return RegionSummary(threshold_value, float(thresholds[region_index, 1]), np.array([0.0, 0.0, threshold_value]), 0.0)

    x = np.arange(tot_map.shape[0], dtype=float)
    x_plot = np.linspace(x[args.plot_range], 2500, 1000)
    tot_err = np.full_like(tot_int, TOT_BIN_ERROR, dtype=float)
    threshold_value = float(thresholds[region_index, 0])
    threshold_sigma = float(thresholds[region_index, 1])

    print(f'{section_name}: threshold={threshold_value:.2f} DAC, sigma={threshold_sigma:.2f} DAC')
    print('tot_int', [f'{value:.2f}' for value in tot_int])
    print('charge_mean', [f'{value:.2f}' for value in charge_mean])
    print('charge_err', [f'{value:.2f}' for value in charge_err])

    if fit_mode == 'fit2par':
        p0 = (7.0, 70.0)
        bounds = ([0.0, 0.0], [np.inf, np.inf])
        popt, pcov = curve_fit(
            lambda charge, a, b: calibration_func(charge, a, b, threshold_value),
            charge_mean,
            tot_int,
            sigma=tot_err,
            p0=p0,
            bounds=bounds,
        )
        fit_curve_params = np.array([popt[0], popt[1], threshold_value])
        displayed_params = fit_curve_params
        free_param_names = ['a', 'b']
        chi2_val, ndof = chi_squared(
            calibration_func,
            charge_mean,
            tot_int,
            tot_err,
            fit_curve_params,
            n_free_params=2,
            label=f'Chi² - {section_name}',
        )
        model_label = '$f_{2par}(x)=(a/x +1/b)*(x-THR)$'
    else:
        p0 = (4.45, 100.0, threshold_value)
        bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(
            calibration_func,
            charge_mean,
            tot_int,
            sigma=tot_err,
            p0=p0,
            bounds=bounds,
        )
        fit_curve_params = np.asarray(popt)
        displayed_params = fit_curve_params
        free_param_names = ['a', 'b', 'd']
        chi2_val, ndof = chi_squared(
            calibration_func,
            charge_mean,
            tot_int,
            tot_err,
            fit_curve_params,
            n_free_params=3,
            label=f'Chi² - {section_name}',
        )
        model_label = '$f_{3par}(x)=(a/x +1/b)*(x-d)$'

    if ndof <= 0:
        chi2_ndof = 0.0
    else:
        chi2_ndof = chi2_val / ndof

    print(f'Fit results {section_name}:', ', '.join(f'{value:.2f}' for value in displayed_params))
    plot_calibration_summary(
        pdf_file=pdf_file,
        section_name=section_name,
        x_plot=x_plot,
        tot_map=tot_map,
        results_clean=results_clean,
        charge_mean=charge_mean,
        tot_int=tot_int,
        fit_curve_params=fit_curve_params,
        displayed_params=displayed_params,
        free_param_names=free_param_names,
        free_cov=pcov,
        chi2_val=chi2_val,
        ndof=max(ndof, 1),
        model_label=model_label,
        cmap=cmap,
    )
    return RegionSummary(threshold_value, threshold_sigma, fit_curve_params, chi2_ndof)


def run_tot_calibration(fit_mode: str) -> None:
    args = build_arg_parser().parse_args()
    input_file = resolve_input_file(args.input_file)
    folder_path = os.path.dirname(input_file)
    file_name = os.path.basename(input_file)
    base_name = os.path.splitext(file_name)[0]

    print(f'[INFO] File selected: {file_name}')
    print(f'[INFO] Folder: {folder_path}')

    with tb.open_file(input_file, 'r') as h5_file:
        config = read_scan_config(h5_file)

    delta_v = compute_delta_v(config)
    regions = make_regions(config)
    hist_arrays = read_hist_arrays(input_file)
    cmap = build_colormap()

    print('Configuration:')
    print(config)
    nmin = (config.stop_column - config.start_column) * (config.stop_row - config.start_row) * config.n_injections / 5.0
    print(f'Nmin={nmin}')

    pdf_name = os.path.join(folder_path, f'{base_name}_tot_calibration_{fit_mode}_charge_mean.pdf')
    region_summaries: List[RegionSummary] = []

    with pdf_backend.PdfPages(pdf_name) as pdf_file:
        plot_scurves(pdf_file, hist_arrays['HistOcc'], delta_v, regions, cmap)
        plot_threshold_map(pdf_file, hist_arrays['ThresholdMap'])
        thresholds = plot_threshold_distributions(pdf_file, hist_arrays['ThresholdMap'], regions, args.center)
        plot_noise_map(pdf_file, hist_arrays['NoiseMap'])
        plot_noise_distributions(pdf_file, hist_arrays['NoiseMap'], regions)

        print('\nplot ToT')
        for region_index, region in enumerate(regions):
            tot_map = build_tot_map(hist_arrays['HistTot'], delta_v, region)
            try:
                summary = fit_region_calibration(
                    section_name=region.label,
                    tot_map=tot_map,
                    thresholds=thresholds,
                    region_index=region_index,
                    args=args,
                    fit_mode=fit_mode,
                    pdf_file=pdf_file,
                    cmap=cmap,
                    nmin=nmin,
                )
            except Exception as exc:
                print(f'[WARNING] Calibration failed for {region.label}: {exc}')
                threshold_value = float(thresholds[region_index, 0])
                threshold_sigma = float(thresholds[region_index, 1])
                summary = RegionSummary(threshold_value, threshold_sigma, np.array([0.0, 0.0, threshold_value]), 0.0)
            region_summaries.append(summary)

    inj_tot_cal = build_inj_tot_calibration(region_summaries)
    output_h5 = write_output_h5(input_file, base_name, fit_mode, inj_tot_cal)

    print(f'[INFO] Created PDF: {pdf_name}')
    print(f'[INFO] Created H5: {output_h5}')
    print(f'THR average: {region_summaries[1].threshold_mean:.2f}')


__all__ = ['run_tot_calibration']
