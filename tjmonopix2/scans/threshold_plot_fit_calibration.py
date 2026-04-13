"""Cleaner entrypoint for threshold-scan ToT calibration.

This script keeps the behavior of the validated calibration workflow, but
organizes the execution flow in a shorter and more readable way.  The heavy
lifting is delegated to the working helper functions already present in
`threshold_plot_fit_calibration_GR_minTOTzero_charge_mean.py`.

The main idea is:
1. Resolve the input file and scan configuration.
2. Select the frontend(s) that were actually scanned.
3. Produce diagnostic PDF pages only for those active frontends.
4. Fit the ToT calibration curve with either a 2-parameter or 3-parameter
   model.
5. Write the resulting `InjTotCalibration` array into a copied HDF5 file.

Keeping this orchestration layer separate makes the workflow easier to read and
modify, while the numerically-validated helper functions remain unchanged.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import matplotlib.backends.backend_pdf as pdf
from matplotlib.backends.backend_pdf import PdfPages

try:
    from tjmonopix2.scans import threshold_plot_fit_calibration_GR_minTOTzero_charge_mean as legacy
except ImportError:
    import threshold_plot_fit_calibration_GR_minTOTzero_charge_mean as legacy


def build_parser():
    """Reuse the validated parser from the working calibration script.

    The parser already exposes the current public interface:
    - input file selection
    - fit mode selection
    - common ToT fit range
    - plot/debug options
    """
    return legacy.build_parser()


def print_run_summary(file_path: str, config: legacy.ScanConfig, delta_v, active_regions) -> None:
    """Print a compact summary of the analysis configuration."""
    print(f'[INFO] File selezionato: {os.path.basename(file_path)}')
    print(f'[INFO] Cartella: {os.path.dirname(file_path)}')
    print('Configuration:')
    print(
        f'cols={config.start_column}:{config.stop_column}, rows={config.start_row}:{config.stop_row}, '
        f'n_inj={config.n_injections}, VH={config.vcal_high}, VL_start={config.vcal_low_start}, '
        f'VL_stop={config.vcal_low_stop}, VL_step={config.vcal_low_step}, nsteps={len(delta_v)}'
    )
    nmin = (config.stop_column - config.start_column) * (config.stop_row - config.start_row) * config.n_injections / 5.0
    print(f'Active frontends: {", ".join(region.label for region in active_regions) if active_regions else "none"}')
    print(f'Nmin={nmin:.1f}')


def create_output_paths(file_path: str, fit_mode: str):
    """Build all output paths derived from the input HDF5 file name."""
    folder_path = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    main_pdf = os.path.join(folder_path, f'{base_name}_tot_calibration_{fit_mode}_charge_mean.pdf')
    debug_pdf = os.path.join(folder_path, f'{base_name}_charge_DAC_tot_distribution_ALL.pdf')
    return main_pdf, debug_pdf


def run_analysis(args) -> tuple[str, str | None, str, dict[str, object]]:
    """Run the full analysis workflow and return created outputs.

    Returns:
    - main PDF path
    - optional charge-debug PDF path
    - output HDF5 path
    - threshold summary dictionary for active frontends
    """
    file_path = legacy.resolve_input_file(args.input_file)
    config = legacy.read_scan_config(file_path)
    delta_v = legacy.compute_delta_v(config)
    all_regions = legacy.make_regions(config)
    active_regions = [region for region in all_regions if legacy.is_region_active(region, config)]
    histograms = legacy.read_histograms(file_path)
    cmap = legacy.build_colormap()
    nmin = (config.stop_column - config.start_column) * (config.stop_row - config.start_row) * config.n_injections / 5.0

    print_run_summary(file_path, config, delta_v, active_regions)

    main_pdf_path, charge_debug_pdf_path = create_output_paths(file_path, args.fit_mode)
    region_calibrations: dict[str, legacy.RegionCalibration] = {}

    # The main PDF collects summary pages, maps and fit plots.
    # The optional debug PDF stores one page per ToT-bin charge distribution.
    with pdf.PdfPages(main_pdf_path) as main_pdf, PdfPages(charge_debug_pdf_path) if args.save_charge_debug_pdf else nullcontext() as debug_pdf:
        legacy.plot_scurves(main_pdf, histograms['HistOcc'], delta_v, active_regions, cmap)
        legacy.plot_threshold_map(main_pdf, histograms['ThresholdMap'], config)
        thresholds = legacy.plot_threshold_distributions(main_pdf, histograms['ThresholdMap'], active_regions, args.verbose)
        legacy.plot_noise_map(main_pdf, histograms['NoiseMap'], config)
        legacy.plot_noise_distributions(main_pdf, histograms['NoiseMap'], active_regions)

        print('\nplot ToT')
        for region_index, region in enumerate(all_regions):
            if region.label not in {active.label for active in active_regions}:
                continue

            tot_map = legacy.build_tot_map(histograms['HistTot'], delta_v, region)
            try:
                calibration = legacy.run_region_fit(
                    pdf_file=main_pdf,
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
                calibration = legacy.RegionCalibration(legacy.np.array([0.0, 0.0, threshold_value]), 0.0)

            region_calibrations[region.label] = calibration

    ordered_calibrations = [
        region_calibrations.get(region_label, legacy.RegionCalibration(legacy.np.zeros(3), 0.0))
        for _, _, region_label in legacy.REGIONS
    ]
    output_array = legacy.build_output_array(ordered_calibrations)
    output_h5 = legacy.write_output_h5(file_path, args.fit_mode, output_array)

    return (
        main_pdf_path,
        charge_debug_pdf_path if args.save_charge_debug_pdf else None,
        output_h5,
        thresholds,
    )


def main() -> None:
    """User-facing entrypoint.

    This function stays intentionally short: it only parses arguments, runs the
    analysis, and prints the final output summary.
    """
    args = build_parser().parse_args()
    main_pdf_path, debug_pdf_path, output_h5, thresholds = run_analysis(args)

    print(f'[INFO] Created PDF: {main_pdf_path}')
    if debug_pdf_path is not None:
        print(f'[INFO] Created charge debug PDF: {debug_pdf_path}')
    print(f'[INFO] Created H5: {output_h5}')
    print('Thresholds (active frontends):', {label: [f'{values[0]:.1f}', f'{values[1]:.1f}'] for label, values in thresholds.items()})


if __name__ == '__main__':
    main()
