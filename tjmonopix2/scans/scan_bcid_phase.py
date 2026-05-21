#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import argparse
import json
import os
import re
from copy import deepcopy

import numpy as np
import tables as tb

from tjmonopix2.analysis import analysis_utils as au
from tjmonopix2.analysis import monitoring
from tjmonopix2.analysis import plotting
from tjmonopix2.scans.scan_threshold import ThresholdScan
from tjmonopix2.scans.shift_and_inject import DEFAULT_PULSE_START_CNFG
from tjmonopix2.system import telegram_bot


scan_configuration = {
    'start_column': 370,
    'stop_column': 372,
    'start_row': 0,
    'stop_row': 512,

    'n_injections': 100,
    'VCAL_HIGH': 140,
    'VCAL_LOW_start': 140 - 0,
    'VCAL_LOW_stop': 140 - 60,
    'VCAL_LOW_step': -1,

    # Sweep several pulse-start settings and compare the threshold response of
    # the same ROI to identify the BCID phase dependence.
    'pulse_start_values': [10, 19, 25, 30, 35, 40, 45, 55],
}

DEFAULT_MANIFEST_FILE = 'scan_bcid_phase_manifest.json'


def parse_pulse_start_values(value):
    values = []
    for token in value.split(','):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError('At least one pulse-start value must be provided.')
    return values


def validate_scan_outputs(raw_file, interpreted_file):
    if not os.path.isfile(raw_file):
        raise RuntimeError(f'Raw output file not found: {raw_file}')
    if not os.path.isfile(interpreted_file):
        raise RuntimeError(f'Interpreted output file not found: {interpreted_file}')

    with tb.open_file(raw_file, mode='r') as h5_file:
        raw_words = h5_file.root.raw_data.nrows
        meta_rows = h5_file.root.meta_data.nrows
    if raw_words == 0 or meta_rows == 0:
        raise RuntimeError(f'No raw data recorded for {os.path.basename(raw_file)}')

    with tb.open_file(interpreted_file, mode='r') as h5_file:
        hist_occ = np.asarray(h5_file.root.HistOcc)
    if hist_occ.size == 0 or np.sum(hist_occ) == 0:
        raise RuntimeError(f'Analysis produced no occupancy for {os.path.basename(interpreted_file)}')


def _mask_disabled_pixels(use_pixel, scan_config):
    mask = np.invert(use_pixel)
    mask[:scan_config['start_column'], :] = True
    mask[scan_config['stop_column']:, :] = True
    mask[:, :scan_config['start_row']] = True
    mask[:, scan_config['stop_row']:] = True
    return mask


def _load_interpreted_metrics(interpreted_file, pulse_start_cnfg):
    with tb.open_file(interpreted_file, mode='r') as h5_file:
        root = h5_file.root
        scan_config = au.ConfigDict(root.configuration_in.scan.scan_config[:])
        run_config = au.ConfigDict(root.configuration_in.scan.run_config[:])
        monitoring_cfg = monitoring.load_monitoring_config_from_root(root)
        threshold_map = root.ThresholdMap[:].astype(np.float32, copy=False)
        noise_map = root.NoiseMap[:].astype(np.float32, copy=False)
        chi2_map = root.Chi2Map[:].astype(np.float32, copy=False)
        use_pixel = root.configuration_in.chip.use_pixel[:].astype(bool, copy=False)

    # Reuse the same notion of "valid fitted pixel" as the standard threshold
    # plots, while restricting the maps to the scanned ROI only.
    disabled_mask = _mask_disabled_pixels(use_pixel, scan_config)
    valid_mask = ~disabled_mask
    valid_mask &= chi2_map > 0.0
    valid_mask &= np.isfinite(threshold_map)
    valid_mask &= np.isfinite(noise_map)
    valid_mask &= threshold_map >= 0.0
    valid_mask &= noise_map >= 0.0

    if not np.any(valid_mask):
        raise RuntimeError(f'No valid threshold-fit pixels found in {os.path.basename(interpreted_file)}')

    threshold_values = threshold_map[valid_mask]
    noise_values = noise_map[valid_mask]

    return {
        'pulse_start_cnfg': int(pulse_start_cnfg),
        'interpreted_file': interpreted_file,
        'scan_config': {str(key): scan_config[key] for key in scan_config.keys()},
        'run_config': {str(key): run_config[key] for key in run_config.keys()},
        'threshold_map': threshold_map,
        'noise_map': noise_map,
        'chi2_map': chi2_map,
        'valid_mask': valid_mask,
        'threshold_mean': float(np.mean(threshold_values)),
        'threshold_std': float(np.std(threshold_values)),
        'noise_mean': float(np.mean(noise_values)),
        'valid_pixel_count': int(np.count_nonzero(valid_mask)),
        'monitoring_cfg': monitoring_cfg,
    }


def _build_run_suffix(pulse_start_cnfg):
    return f'_bcid_pulse{pulse_start_cnfg}'


def _build_summary_base_path(first_raw_file):
    base_path = first_raw_file[:-3] if first_raw_file.endswith('.h5') else first_raw_file
    return re.sub(r'_bcid_pulse-?\d+$', '', base_path)


def _build_output_path(base_path, suffix):
    candidate = f'{base_path}_{suffix}'
    if not os.path.exists(candidate):
        return candidate

    index = 2
    while True:
        numbered = f'{base_path}_{suffix}_{index}'
        if not os.path.exists(numbered):
            return numbered
        index += 1


def _default_manifest_path(summary_base):
    return f'{summary_base}_{DEFAULT_MANIFEST_FILE}'


def _build_summary_payload(scan_results):
    if not scan_results:
        raise RuntimeError('No valid threshold scans are available for BCID summary plotting.')

    threshold_means = np.array([result['threshold_mean'] for result in scan_results], dtype=float)
    min_idx = int(np.argmin(threshold_means))
    max_idx = int(np.argmax(threshold_means))
    min_result = scan_results[min_idx]
    max_result = scan_results[max_idx]

    common_valid_mask = None
    delta_threshold_map = None
    threshold_effective_map = None
    noise_effective_map = None
    has_effective_maps = False

    # Effective maps are only meaningful if at least two pulse-start points
    # produced usable threshold/noise fits.
    if len(scan_results) >= 2:
        common_valid_mask = min_result['valid_mask'] & max_result['valid_mask']
        if np.any(common_valid_mask):
            delta_threshold_map = np.full_like(min_result['threshold_map'], np.nan, dtype=np.float32)
            threshold_effective_map = np.full_like(min_result['threshold_map'], np.nan, dtype=np.float32)
            noise_effective_map = np.full_like(min_result['noise_map'], np.nan, dtype=np.float32)

            delta_threshold = max_result['threshold_map'][common_valid_mask] - min_result['threshold_map'][common_valid_mask]
            threshold_effective = min_result['threshold_map'][common_valid_mask] + delta_threshold / 2.0
            noise_mean = 0.5 * (
                min_result['noise_map'][common_valid_mask] +
                max_result['noise_map'][common_valid_mask]
            )
            noise_effective = np.sqrt(np.square(noise_mean) + np.square(delta_threshold / 2.0))

            delta_threshold_map[common_valid_mask] = delta_threshold
            threshold_effective_map[common_valid_mask] = threshold_effective
            noise_effective_map[common_valid_mask] = noise_effective
            has_effective_maps = True

    return {
        'scan_results': scan_results,
        'pulse_start_values': [result['pulse_start_cnfg'] for result in scan_results],
        'min_threshold_result': min_result,
        'max_threshold_result': max_result,
        'delta_threshold_map': delta_threshold_map,
        'threshold_effective_map': threshold_effective_map,
        'noise_effective_map': noise_effective_map,
        'common_valid_mask': common_valid_mask,
        'has_effective_maps': has_effective_maps,
        'monitoring_tables': _load_full_sequence_monitoring(scan_results),
        'scan_config': min_result['scan_config'],
        'run_config': min_result['run_config'],
    }


def _get_combined_scan_window(raw_files):
    scan_start = None
    scan_stop = None
    for raw_file in raw_files:
        with tb.open_file(raw_file, mode='r') as h5_file:
            meta = h5_file.root.meta_data[:]
            if meta.size == 0:
                continue
            file_start = float(np.min(meta['timestamp_start']))
            file_stop = float(np.max(meta['timestamp_stop']))
            scan_start = file_start if scan_start is None else min(scan_start, file_start)
            scan_stop = file_stop if scan_stop is None else max(scan_stop, file_stop)
    return scan_start, scan_stop


def _load_full_sequence_monitoring(scan_results):
    if not scan_results:
        return {}

    monitoring_cfg = scan_results[0].get('monitoring_cfg')
    if not monitoring_cfg or not monitoring_cfg.get('enable', False):
        return {}

    scan_start, scan_stop = _get_combined_scan_window([result['raw_file'] for result in scan_results])
    if scan_start is None or scan_stop is None:
        return {}

    time_margin = float(monitoring_cfg.get('time_margin_s', 0.0) or 0.0)
    scan_start -= time_margin
    scan_stop += time_margin

    # Read monitoring directly from the CSV sources so the final BCID plots also
    # include the gaps between individual threshold scans.
    merged = {}
    defaults = {
        'power': (
            'timestamp',
            {
                'HV_V': 'HV_V',
                'HV_I': 'HV_I',
                'PWELL_V': 'PWELL_V',
                'PWELL_I': 'PWELL_I',
                'PSUB_PWELL_V': 'PSUB_PWELL_V',
                'PSUB_PWELL_I': 'PSUB_PWELL_I',
            },
        ),
        'env': (
            'Time [s]',
            {
                'NTC_C': 'NTC [°C]',
            },
        ),
    }

    for table_name in ('power', 'env'):
        cfg = monitoring_cfg.get(table_name) or {}
        if not cfg:
            continue
        default_timestamp, default_columns = defaults[table_name]
        ts_col, columns_map = monitoring._prepare_columns(cfg, default_timestamp=default_timestamp, default_columns=default_columns)
        paths = monitoring._iter_csv_paths(cfg)
        data = monitoring._read_csv_data(paths, ts_col, columns_map, scan_start, scan_stop)
        if not data:
            continue
        ts_arr, series_arr, labels = data
        fields = [('timestamp', np.float64)]
        label_map = {}
        unit_map = {}
        for key in series_arr.keys():
            field = monitoring._sanitize_field_name(key)
            label_map[field] = labels.get(key, key)
            unit_map[field] = monitoring._extract_unit(labels.get(key, key), fallback_name=key)
            fields.append((field, np.float64))
        table = np.zeros(shape=(len(ts_arr),), dtype=np.dtype(fields))
        table['timestamp'] = ts_arr
        for key, values in series_arr.items():
            table[monitoring._sanitize_field_name(key)] = values
        merged[table_name] = {
            'data': table,
            'label_map': label_map,
            'unit_map': unit_map,
        }

    return merged


def _summary_payload_to_manifest(payload, requested_pulse_start_values=None, stopped_early=False, stop_reason=None):
    # Persist only the lightweight bookkeeping needed to rebuild the BCID
    # summary later, without duplicating the heavy map arrays in JSON.
    return {
        'requested_pulse_start_values': requested_pulse_start_values or payload['pulse_start_values'],
        'pulse_start_values': payload['pulse_start_values'],
        'min_threshold_pulse_start': payload['min_threshold_result']['pulse_start_cnfg'],
        'max_threshold_pulse_start': payload['max_threshold_result']['pulse_start_cnfg'],
        'has_effective_maps': payload.get('has_effective_maps', False),
        'stopped_early': stopped_early,
        'stop_reason': stop_reason,
        'scan_config': payload.get('scan_config', {}),
        'run_config': payload.get('run_config', {}),
        'summary_pdf': payload.get('summary_pdf'),
        'scan_results': [
            {
                'pulse_start_cnfg': result['pulse_start_cnfg'],
                'raw_file': result['raw_file'],
                'interpreted_file': result['interpreted_file'],
                'threshold_mean': result['threshold_mean'],
                'threshold_std': result['threshold_std'],
                'noise_mean': result['noise_mean'],
                'valid_pixel_count': result['valid_pixel_count'],
            }
            for result in payload['scan_results']
        ],
    }

def _write_manifest_json(manifest_path, payload, requested_pulse_start_values=None, stopped_early=False, stop_reason=None):
    manifest = _summary_payload_to_manifest(
        payload,
        requested_pulse_start_values=requested_pulse_start_values,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w', encoding='ascii') as manifest_file:
        json.dump(manifest, manifest_file, indent=2)


def _load_metrics_from_manifest(manifest_path):
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f'Manifest file not found: {manifest_path}')
    with open(manifest_path, 'r', encoding='ascii') as manifest_file:
        manifest = json.load(manifest_file)
    scan_results = []
    for entry in sorted(manifest.get('scan_results', []), key=lambda item: int(item['pulse_start_cnfg'])):
        pulse_start_cnfg = int(entry['pulse_start_cnfg'])
        raw_file = entry['raw_file']
        interpreted_file = entry['interpreted_file']
        try:
            validate_scan_outputs(raw_file, interpreted_file)
            metrics = _load_interpreted_metrics(interpreted_file, pulse_start_cnfg)
            metrics['raw_file'] = raw_file
            scan_results.append(metrics)
        except Exception:
            continue
    return scan_results, manifest


def _threshold_scan_config_for_pulse(pulse_start_cnfg):
    # Reuse the regular threshold scan implementation, changing only the pulse
    # timing for the current BCID scan step.
    threshold_config = deepcopy(scan_configuration)
    threshold_config.pop('pulse_start_values', None)
    threshold_config['pulse_start_cnfg'] = pulse_start_cnfg
    return threshold_config


def _prompt_recovery_action(pulse_start_cnfg, error_message):
    print()
    print(f'PulseStartCnfg {pulse_start_cnfg} failed: {error_message}')
    print('You can now power-cycle the chip and choose what to do next.')
    print("Type 'r' to retry the same pulse start after recovery.")
    print("Type 'q' to stop here and create a partial summary from the valid scans.")
    while True:
        try:
            choice = input('Recovery action [r/q]: ').strip().lower()
        except KeyboardInterrupt:
            print()
            return 'stop'
        if choice in {'r', 'retry'}:
            return 'retry'
        if choice in {'q', 'quit', 'stop'}:
            return 'stop'
        print("Please type 'r' or 'q'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pulse-start-values", type=parse_pulse_start_values, default=None)
    parser.add_argument("--manifest-file", type=str, default=None, help="Path to a BCID summary manifest JSON file.")
    parser.add_argument("--plot-only", action="store_true", help="Skip data taking and build the BCID summary from an existing manifest file.")
    parser.add_argument("--def_regs", help="Force the use of default registers", action="store_true")
    parser.add_argument("--regs_json", type=str, help="Path to JSON file with regs configs", default="../chip_registers.json")
    parser.add_argument("--chip", type=str, help="Chip name (e.g., W8R6)")
    parser.add_argument("--fe", type=str, help="FE name (e.g., HVC or DCC)")
    parser.add_argument("--h5_config_file", type=str, default=None)
    args = parser.parse_args()

    pulse_start_values = args.pulse_start_values or list(scan_configuration['pulse_start_values'])
    scan_results = []
    stopped_early = False
    stop_reason = None
    manifest_path = os.path.abspath(args.manifest_file) if args.manifest_file else None

    if args.plot_only:
        # Rebuild the BCID summary from previously recorded scan outputs without
        # taking new data.
        if manifest_path is None:
            raise RuntimeError('Please provide --manifest-file to rebuild the BCID summary plots.')
        scan_results, manifest = _load_metrics_from_manifest(manifest_path)
        if not scan_results:
            raise RuntimeError(f'No valid BCID threshold scans found in manifest: {manifest_path}')
        pulse_start_values = manifest.get('requested_pulse_start_values', pulse_start_values)
        stopped_early = bool(manifest.get('stopped_early', False))
        stop_reason = manifest.get('stop_reason')
    else:
        for pulse_start_cnfg in pulse_start_values:
            run_scan_config = _threshold_scan_config_for_pulse(pulse_start_cnfg)
            if args.h5_config_file:
                run_scan_config["chip_config_file"] = args.h5_config_file

            while True:
                try:
                    # Each pulse-start value is executed as an independent
                    # threshold scan so failures are easy to isolate and retry.
                    with ThresholdScan(scan_config=run_scan_config, suffix=_build_run_suffix(pulse_start_cnfg)) as scan:
                        if args.def_regs:
                            scan.configuration.setdefault("configure", {})
                            scan.configuration["configure"].update({
                                "def_regs": args.def_regs,
                                "regs_json": args.regs_json,
                                "chip": args.chip,
                                "fe": args.fe
                            })
                        scan.start()
                        raw_file = scan.output_filename + ".h5"
                        interpreted_file = scan.output_filename + "_interpreted.h5"

                    validate_scan_outputs(raw_file, interpreted_file)
                    metrics = _load_interpreted_metrics(interpreted_file, pulse_start_cnfg)
                    metrics['raw_file'] = raw_file
                    scan_results.append(metrics)
                    break
                except KeyboardInterrupt:
                    # If the user interrupts after at least one successful scan,
                    # the script still falls through to partial-summary creation.
                    stopped_early = True
                    stop_reason = f'Scan interrupted by user while running pulse_start_cnfg={pulse_start_cnfg}'
                    break
                except Exception as error:
                    error_text = str(error)
                    action = _prompt_recovery_action(pulse_start_cnfg, error_text)
                    if action == 'retry':
                        print(f'Retrying PulseStartCnfg {pulse_start_cnfg}...')
                        continue
                    stopped_early = True
                    stop_reason = f'PulseStartCnfg {pulse_start_cnfg} failed and the run was stopped by the user.'
                    break

            if stopped_early:
                break

    if not scan_results:
        raise RuntimeError('No valid BCID threshold scans available for BCID summary plotting.')

    scan_results = sorted(scan_results, key=lambda result: result['pulse_start_cnfg'])
    summary_payload = _build_summary_payload(scan_results)
    summary_base = _build_summary_base_path(scan_results[0]['raw_file'])
    if manifest_path is None:
        manifest_path = _default_manifest_path(summary_base)
    # Derive a fresh output name so rerunning the summary does not overwrite a
    # previous PDF unless the caller explicitly chooses the same destination.
    summary_pdf = _build_output_path(summary_base, 'bcid_phase_summary.pdf')
    summary_payload['summary_pdf'] = summary_pdf
    plotting.create_bcid_phase_summary_pdf(summary_payload, summary_pdf)
    _write_manifest_json(
        manifest_path,
        summary_payload,
        requested_pulse_start_values=pulse_start_values,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )
    if stopped_early:
        telegram_bot.send_message_generic(
            telegram_bot.THREAD_ID_SCANS,
            f'bcid_phase_scan stopped early; partial summary created. {stop_reason}',
            summary_pdf,
        )
        print(f'BCID phase scan stopped early. Partial summary PDF written to {summary_pdf}')
        if stop_reason:
            print(stop_reason)
    else:
        telegram_bot.send_message_generic(
            telegram_bot.THREAD_ID_SCANS,
            'bcid_phase_scan completed! 🦆',
            summary_pdf,
        )
        print(f'BCID phase scan finished. Summary PDF written to {summary_pdf}')
    print(f'BCID phase scan manifest written to {manifest_path}')
