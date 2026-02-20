import os
import pandas as pd
import cffwis as fwi


def calc_ffmc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Fine Fuel Moisture Code (FFMC) for each row in the DataFrame.
    The calculation uses the previous day's FFMC as input for the next day's calculation.
    The result is stored in a new column 'FFMC'.
    Args:
        df (pd.DataFrame): DataFrame containing columns 'dailyFineFuelMoistureCode', 'TEMP', 'RH', 'WS', 'PCP_ACCUM'.
    Returns:
        pd.DataFrame: DataFrame with an added 'FFMC' column.
    """
    # Calculate FFMC for each row and store in a list
    ffmc_values = []
    ffmc0 = None
    for count, (idx, row) in enumerate(df.iterrows()):
        if count == 0:
            ffmc = row['dailyFineFuelMoistureCode']
        else:
            # Calculate FFMC
            ffmc = fwi.dailyFFMC(
                ffmc0,
                row['TEMP'],
                row['RH'],
                row['WS'],
                row['PCP_ACCUM']
            )
        ffmc_values.append(ffmc)
        ffmc0 = ffmc  # update for next iteration

    # Assign FFMC values
    df['FFMC'] = ffmc_values

    return df


def calc_dmc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Duff Moisture Code (DMC) for each row in the DataFrame.
    The calculation uses the previous day's DMC as input for the next day's calculation.
    The result is stored in a new column 'DMC'.
    Args:
        df (pd.DataFrame): DataFrame containing columns 'dailyDuffMoistureCode', 'TEMP', 'RH', 'PCP_ACCUM', 'month'.
    Returns:
        pd.DataFrame: DataFrame with an added 'DMC' column.
    """
    # Calculate DMC for each row and store in a list
    dmc_values = []
    dmc0 = None
    for count, (idx, row) in enumerate(df.iterrows()):
        if count == 0:
            dmc = row['dailyDuffMoistureCode']
        else:
            dmc = fwi.dailyDMC(
                dmc0,
                row['TEMP'],
                row['RH'],
                row['PCP_ACCUM'],
                row['month']
            )
        dmc_values.append(dmc)
        dmc0 = dmc  # update for next iteration

    # Assign DMC values
    df['DMC'] = dmc_values

    return df


def calc_dc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Drought Code (DC) for each row in the DataFrame.
    The calculation uses the previous day's DC as input for the next day's calculation.
    The result is stored in a new column 'DC'.
    Args:
        df (pd.DataFrame): DataFrame containing columns 'dailyDroughtCode', 'TEMP', 'PCP_ACCUM', 'month'.
    Returns:
        pd.DataFrame: DataFrame with an added 'DC' column.
    """
    # Calculate DC for each row and store in a list
    dc_values = []
    dc0 = None
    for count, (idx, row) in enumerate(df.iterrows()):
        if count == 0:
            dc = row['dailyDroughtCode']
        else:
            dc = fwi.dailyDC(
                dc0,
                row['TEMP'],
                row['PCP_ACCUM'],
                row['month']
            )
        dc_values.append(dc)
        dc0 = dc  # update for next iteration

    # Assign DC values
    df['DC'] = dc_values

    return df


def calc_isi_generic(df: pd.DataFrame, ws_col: str, ffmc_col: str, out_col: str) -> pd.DataFrame:
    """
    Calculate the Initial Spread Index (ISI) for the given DataFrame using specified wind speed and FFMC columns.
    The result is stored in the column specified by out_col.
    Args:
        df (pd.DataFrame): DataFrame containing wind speed and FFMC columns.
        ws_col (str): Name of the wind speed column.
        ffmc_col (str): Name of the FFMC column.
        out_col (str): Name of the output ISI column.
    Returns:
        pd.DataFrame: DataFrame with the ISI column added.
    """
    df[out_col] = fwi.dailyISI(
        df[ws_col].values,
        df[ffmc_col].values,
    )
    return df


def calc_bui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Buildup Index (BUI) for each row in the DataFrame using DMC and DC columns.
    The result is stored in a new column 'BUI'.
    Args:
        df (pd.DataFrame): DataFrame containing columns 'DMC' and 'DC'.
    Returns:
        pd.DataFrame: DataFrame with an added 'BUI' column.
    """
    df['BUI'] = fwi.dailyBUI(
        df['DMC'].values,
        df['DC'].values,
    )
    return df


def calc_fwi_generic(df: pd.DataFrame, isi_col: str, bui_col: str, out_col: str) -> pd.DataFrame:
    """
    Calculate the Fire Weather Index (FWI) for the given DataFrame using specified ISI and BUI columns.
    The result is stored in the column specified by out_col.
    Args:
        df (pd.DataFrame): DataFrame containing ISI and BUI columns.
        isi_col (str): Name of the ISI column.
        bui_col (str): Name of the BUI column.
        out_col (str): Name of the output FWI column.
    Returns:
        pd.DataFrame: DataFrame with the FWI column added.
    """
    df[out_col] = fwi.dailyFWI(
        df[isi_col].values,
        df[bui_col].values,
    )
    return df


def calc_hffmc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Hourly Fine Fuel Moisture Code (hFFMC) for each row in the DataFrame.
    The calculation uses the previous hour's hFFMC as input for the next hour's calculation.
    The result is stored in a new column 'hFFMC'.
    Args:
        df (pd.DataFrame): DataFrame containing columns 'hourlyFineFuelMoistureCode', 'TEMP', 'RH', 'WS', 'PCP'.
    Returns:
        pd.DataFrame: DataFrame with an added 'hFFMC' column.
    """
    # Calculate HFFMC for each row and store in a list
    hffmc_values = []
    hffmc0 = None
    for count, (idx, row) in enumerate(df.iterrows()):
        if count == 0:
            hffmc = row['hourlyFineFuelMoistureCode']
        else:
            hffmc = fwi.hourlyFFMC(
                hffmc0,
                row['TEMP'],
                row['RH'],
                row['WS'],
                row['PCP'],
                use_precise_values=True,
            )
        hffmc_values.append(hffmc)
        hffmc0 = hffmc  # update for next iteration

    # Assign HFFMC values
    df['hFFMC'] = hffmc_values

    return df


if __name__ == '__main__':
    # Define input and output folders
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    inputs_dir = os.path.join(data_dir, 'inputs')
    outputs_dir = os.path.join(data_dir, 'outputs')

    # Ensure the output folder exists
    os.makedirs(outputs_dir, exist_ok=True)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(inputs_dir, 'haig_camp_bcws_stn_2023.csv'))

    # Calculate month from weatherTimestamp (yyyymmddhh) formatted integer
    df['month'] = df['weatherTimestamp'].astype(str).str[4:6].astype(int)

    # ### RUN DAILY CALCULATIONS ###
    print('\nProcessing Haig Camp Daily Weather Data')
    daily_df = df.dropna(subset=['dailyFineFuelMoistureCode']).copy()
    daily_df = calc_ffmc(daily_df)
    daily_df = calc_dmc(daily_df)
    daily_df = calc_dc(daily_df)
    daily_df = calc_isi_generic(daily_df, ws_col='WS', ffmc_col='FFMC', out_col='ISI')
    daily_df = calc_bui(daily_df)
    daily_df = calc_fwi_generic(daily_df, isi_col='ISI', bui_col='BUI', out_col='FWI')
    # Save the processed DataFrame to a new CSV file
    daily_df.to_csv(os.path.join(outputs_dir, 'HaigCamp_daily_weather_results.csv'), index=False)

    # ### RUN HOURLY CALCULATIONS ###
    print('Processing Haig Camp Hourly Weather Data')
    hourly_df = df.copy()

    # Assign daily BUI to hourly df for HFWI calculation
    # Match on date (weatherTimestamp is in the same format in both dfs)
    hourly_df = hourly_df.merge(
        daily_df[['weatherTimestamp', 'BUI']],
        on='weatherTimestamp',
        how='left'
    )
    hourly_df['BUI'] = hourly_df['BUI'].ffill() # forward fill BUI values for hourly rows

    # Drop first set of rows until valid values for hourly FFMC are present
    # (i.e., drop head until first non-NA value in hourlyFineFuelMoistureCode)
    first_valid_index = hourly_df['hourlyFineFuelMoistureCode'].first_valid_index()
    hourly_df = hourly_df.loc[first_valid_index:].copy()

    # Calculate hourly FFMC, HISI, and HFWI
    hourly_df = calc_hffmc(hourly_df)
    hourly_df = calc_isi_generic(hourly_df, ws_col='WS', ffmc_col='hFFMC', out_col='hISI')
    hourly_df = calc_fwi_generic(hourly_df, isi_col='hISI', bui_col='BUI', out_col='hFWI')
    # Save the processed DataFrame to a new CSV file
    hourly_df.to_csv(os.path.join(outputs_dir, 'HaigCamp_hourly_weather_results.csv'), index=False)

    print('Processing complete. Output saved to:', outputs_dir)

