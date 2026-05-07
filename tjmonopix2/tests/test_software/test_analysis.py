import os
import tempfile
import unittest

import numpy as np
import tables as tb

from tjmonopix2.analysis.analysis import Analysis


CONFIG_DTYPE = np.dtype([('attribute', 'S64'), ('value', 'S256')])
META_DTYPE = np.dtype([
    ('index_start', '<i8'),
    ('index_stop', '<i8'),
    ('data_length', '<u4'),
    ('timestamp_start', '<f8'),
    ('timestamp_stop', '<f8'),
    ('scan_param_id', '<u4'),
    ('error', '<u4'),
    ('trigger', '<f8'),
])


def _bin2gray(value):
    return value ^ (value >> 1)


def _tj_word(d0, d1, d2):
    return np.uint32(0x40000000 | ((d0 & 0x1ff) << 18) | ((d1 & 0x1ff) << 9) | (d2 & 0x1ff))


def _timestamp_words(timestamp):
    return [
        np.uint32(0x4c000000 | ((timestamp >> 26) & 0x03ffffff)),
        np.uint32(0x48000000 | (timestamp & 0x03ffffff)),
    ]


def _hit_words(col, row, le, te, timestamp):
    le_gray = _bin2gray(le)
    te_gray = _bin2gray(te)
    return _timestamp_words(timestamp) + [
        _tj_word(0x1bc, (col >> 1) & 0xff, ((le_gray & 0x7f) << 1) | ((te_gray >> 6) & 0x1)),
        _tj_word(((te_gray & 0x3f) << 2) | ((col & 0x1) << 1) | ((row >> 8) & 0x1), row & 0xff, 0x17c),
    ]


def _write_config_table(h5_file, group, name, values):
    table = h5_file.create_table(group, name=name, description=CONFIG_DTYPE)
    for key, value in values.items():
        row = table.row
        row['attribute'] = key.encode()
        row['value'] = repr(value).encode()
        row.append()
    table.flush()


def _create_raw_file(file_path):
    words_scan0 = _hit_words(col=3, row=5, le=10, te=15, timestamp=100)
    words_scan1 = _hit_words(col=3, row=5, le=10, te=18, timestamp=200)
    raw_words = np.array(words_scan0 + words_scan1, dtype=np.uint32)

    with tb.open_file(file_path, mode='w', title='analysis_test') as h5_file:
        h5_file.create_array(h5_file.root, name='raw_data', obj=raw_words)

        meta = np.zeros(2, dtype=META_DTYPE)
        meta[0]['index_start'] = 0
        meta[0]['index_stop'] = len(words_scan0)
        meta[0]['data_length'] = len(words_scan0)
        meta[0]['timestamp_start'] = 0.0
        meta[0]['timestamp_stop'] = 1.0
        meta[0]['scan_param_id'] = 0
        meta[1]['index_start'] = len(words_scan0)
        meta[1]['index_stop'] = len(raw_words)
        meta[1]['data_length'] = len(words_scan1)
        meta[1]['timestamp_start'] = 1.0
        meta[1]['timestamp_stop'] = 2.0
        meta[1]['scan_param_id'] = 1
        meta_table = h5_file.create_table(h5_file.root, name='meta_data', description=META_DTYPE)
        meta_table.append(meta)
        meta_table.flush()

        for cfg_name in ('configuration_in', 'configuration_out'):
            cfg = h5_file.create_group(h5_file.root, cfg_name)
            scan = h5_file.create_group(cfg, 'scan')
            chip = h5_file.create_group(cfg, 'chip')
            bench = h5_file.create_group(cfg, 'bench')

            _write_config_table(h5_file, scan, 'run_config', {
                'scan_id': 'analysis_test',
                'chip_sn': 'test_chip',
            })
            _write_config_table(h5_file, scan, 'scan_config', {
                'start_column': 0,
                'stop_column': 4,
                'start_row': 0,
                'stop_row': 8,
                'n_injections': 1,
            })
            _write_config_table(h5_file, chip, 'settings', {
                'chip_sn': 'test_chip',
            })
            _write_config_table(h5_file, bench, 'TLU', {
                'DATA_FORMAT': 1,
            })


class TestAnalysis(unittest.TestCase):
    def test_analysis_writes_histogram_slices_without_full_scan_param_histograms(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_file = os.path.join(tmp_dir, 'synthetic_scan.h5')
            interpreted_file = os.path.join(tmp_dir, 'synthetic_scan_interpreted.h5')
            _create_raw_file(raw_file)

            with Analysis(raw_data_file=raw_file, analyzed_data_file=interpreted_file, cluster_hits=False) as analyzer:
                analyzer.analyze_data()

            with tb.open_file(interpreted_file, mode='r') as h5_file:
                self.assertEqual(h5_file.root.HistOcc.shape, (512, 512, 2))
                self.assertEqual(h5_file.root.HistTot.shape, (512, 512, 2, 128))

                hist_occ = h5_file.root.HistOcc
                hist_tot = h5_file.root.HistTot
                self.assertEqual(hist_occ[3, 5, 0], 1)
                self.assertEqual(hist_occ[3, 5, 1], 1)
                self.assertEqual(hist_occ[:, :, 0].sum(), 1)
                self.assertEqual(hist_occ[:, :, 1].sum(), 1)

                self.assertEqual(hist_tot[3, 5, 0, 5], 1)
                self.assertEqual(hist_tot[3, 5, 1, 8], 1)
                self.assertEqual(hist_tot[:, :, 0, :].sum(), 1)
                self.assertEqual(hist_tot[:, :, 1, :].sum(), 1)

                dut = h5_file.root.Dut[:]
                self.assertEqual(dut.shape[0], 2)
                self.assertEqual(dut['scan_param_id'].tolist(), [0, 1])

    @unittest.skipUnless(
        os.environ.get('TJMONOPIX2_TEST_RAW_H5'),
        'Set TJMONOPIX2_TEST_RAW_H5 to run analysis on a real raw H5 file.',
    )
    def test_analysis_real_raw_file_smoke(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_file = os.environ['TJMONOPIX2_TEST_RAW_H5']
            interpreted_file = os.path.join(tmp_dir, 'real_scan_interpreted.h5')

            with Analysis(raw_data_file=raw_file, analyzed_data_file=interpreted_file, cluster_hits=False) as analyzer:
                analyzer.analyze_data()

            with tb.open_file(interpreted_file, mode='r') as h5_file:
                self.assertEqual(h5_file.root.HistOcc.shape[:2], (512, 512))
                self.assertEqual(h5_file.root.HistTot.shape[:2], (512, 512))
                self.assertEqual(h5_file.root.HistTot.shape[2], h5_file.root.HistOcc.shape[2])
                self.assertEqual(h5_file.root.HistTot.shape[3], 128)


if __name__ == '__main__':
    unittest.main()
