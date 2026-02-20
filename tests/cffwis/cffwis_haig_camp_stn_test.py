import os
import pandas as pd
import cffwis as fwi


def calc_ffmc(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate FFMC for each row and store in a list
    ffmc_values = []
    for count, row in enumerate(df.itertuples(index=False)):
        if count == 0:
            ffmc = row.dailyFineFuelMoistureCode
        else:
            # Calculate FFMC
            ffmc = fwi.dailyFFMC(
                ffmc0,
                row.TEMP,
                row.RH,
                row.WS,
                row.PCP_ACCUM
            )
        ffmc_values.append(ffmc)
        ffmc0 = ffmc  # update for next iteration

    # Assign FFMC values
    df['FFMC'] = ffmc_values

    return df


def calc_dmc(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate DMC for each row and store in a list
    dmc_values = []
    for count, row in enumerate(df.itertuples(index=False)):
        if count == 0:
            dmc = row.dailyDuffMoistureCode
        else:
            dmc = fwi.dailyDMC(
                dmc0,
                row.TEMP,
                row.RH,
                row.PCP_ACCUM,
                row.month
            )
        dmc_values.append(dmc)
        dmc0 = dmc  # update for next iteration

    # Assign DMC values
    df['DMC'] = dmc_values

    return df


def calc_dc(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate DC for each row and store in a list
    dc_values = []
    for count, row in enumerate(df.itertuples(index=False)):
        if count == 0:
            dc = row.dailyDroughtCode
        else:
            dc = fwi.dailyDC(
                dc0,
                row.TEMP,
                row.PCP_ACCUM,
                row.month
            )
        dc_values.append(dc)
        dc0 = dc  # update for next iteration

    # Assign DC values
    df['DC'] = dc_values

    return df


def calc_isi(df: pd.DataFrame) -> pd.DataFrame:
    df['ISI'] = fwi.dailyISI(
        df['WS'].values,
        df['FFMC'].values,
    )
    return df


def calc_bui(df: pd.DataFrame) -> pd.DataFrame:
    df['BUI'] = fwi.dailyBUI(
        df['DMC'].values,
        df['DC'].values,
    )
    return df


def calc_fwi(df: pd.DataFrame) -> pd.DataFrame:
    df['FWI'] = fwi.dailyFWI(
        df['ISI'].values,
        df['BUI'].values,
    )
    return df


def calc_hffmc(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate HFFMC for each row and store in a list
    hffmc_values = []
    for count, row in enumerate(df.itertuples(index=False)):
        if count == 0:
            hffmc = row.hourlyFineFuelMoistureCode
        else:
            hffmc = fwi.hourlyFFMC(
                hffmc0,
                row.TEMP,
                row.RH,
                row.WS,
                row.PCP,
                use_precise_values=True,
            )
        hffmc_values.append(hffmc)
        hffmc0 = hffmc  # update for next iteration

    # Assign HFFMC values
    df['hFFMC'] = hffmc_values

    return df


def calc_hisi(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate HISI
    df['hISI'] = fwi.dailyISI(
        df['WS'].values,
        df['hFFMC'].values,
    )
    return df


def calc_hfwi(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate HFWI
    df['hFWI'] = fwi.dailyFWI(
        df['hISI'].values,
        df['BUI'].values,
    )
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
    daily_df = calc_isi(daily_df)
    daily_df = calc_bui(daily_df)
    daily_df = calc_fwi(daily_df)
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
    hourly_df = calc_hisi(hourly_df)
    hourly_df = calc_hfwi(hourly_df)
    # Save the processed DataFrame to a new CSV file
    hourly_df.to_csv(os.path.join(outputs_dir, 'HaigCamp_hourly_weather_results.csv'), index=False)


    # ### VALIDATE RESULTS ###

    print('Processing complete. Output saved to:', outputs_dir)
