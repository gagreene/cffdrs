import os
import sys
import subprocess
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def load_output_and_golden(output_path, golden_path, index_col=None, sort_by=None):
    df_out = pd.read_csv(output_path)
    df_golden = pd.read_csv(golden_path)
    if sort_by:
        df_out = df_out.sort_values(by=sort_by).reset_index(drop=True)
        df_golden = df_golden.sort_values(by=sort_by).reset_index(drop=True)
    if index_col:
        df_out = df_out.set_index(index_col)
        df_golden = df_golden.set_index(index_col)
    return df_out, df_golden


data_dir = os.path.join(os.path.dirname(__file__), 'cffwis', 'data')
golden_output_dir = os.path.join(data_dir, 'golden_outputs')
outputs_dir = os.path.join(data_dir, 'outputs')

# Ensure outputs are generated before running tests
@pytest.fixture(scope='module', autouse=True)
def generate_outputs():
    # Check if both output files exist
    daily_output = os.path.join(outputs_dir, 'HaigCamp_daily_weather_results.csv')
    hourly_output = os.path.join(outputs_dir, 'HaigCamp_hourly_weather_results.csv')
    if not (os.path.exists(daily_output) and os.path.exists(hourly_output)):
        script_path = os.path.join(os.path.dirname(__file__), 'cffwis', 'cffwis_haig_camp_stn.py')
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        assert result.returncode == 0, f'Script failed: {result.stderr}'


test_cases = [
    (
        os.path.join(outputs_dir, 'HaigCamp_daily_weather_results.csv'),
        os.path.join(golden_output_dir, 'HaigCamp_daily_weather_results.csv'),
        'weatherTimestamp',
    ),
    (
        os.path.join(outputs_dir, 'HaigCamp_hourly_weather_results.csv'),
        os.path.join(golden_output_dir, 'HaigCamp_hourly_weather_results.csv'),
        'weatherTimestamp',
    ),
]

@pytest.mark.parametrize('output_path,golden_path,index_col', test_cases)
def test_outputs_match_golden(output_path, golden_path, index_col):
    df_out, df_golden = load_output_and_golden(output_path, golden_path, index_col=index_col, sort_by=index_col)
    common_cols = [col for col in df_out.columns if col in df_golden.columns]
    for col in common_cols:
        if pd.api.types.is_float_dtype(df_out[col]):
            df_out[col] = df_out[col].astype('float64')
        if pd.api.types.is_float_dtype(df_golden[col]):
            df_golden[col] = df_golden[col].astype('float64')
    try:
        assert_frame_equal(
            df_out[common_cols],
            df_golden[common_cols],
            check_dtype=False,
            check_like=True,
            rtol=1e-8,
            atol=1e-10
        )
    except AssertionError as e:
        print('Column dtypes:')
        print(df_out[common_cols].dtypes)
        print(df_golden[common_cols].dtypes)
        print('Sample values from hISI:')
        print('Output:', df_out['hISI'].head(10).tolist())
        print('Golden:', df_golden['hISI'].head(10).tolist())
        raise
