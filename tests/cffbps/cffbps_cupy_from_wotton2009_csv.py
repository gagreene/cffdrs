from cffbps_cupy import FBP
import os
import pandas as pd
import numpy as np

# Fuel types
fuel_type_lookup = {
    'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
    'D1': 8, 'D2': 9,
    'M1': 10, 'M2': 11, 'M3': 12, 'M4': 13,
    'O1A': 14, 'O1B': 15,
    'S1': 16, 'S2': 17, 'S3': 18,
    'NF': 19, 'WA': 20
}

# Set directory
# You can change this to your working directory by typing data_dir =  r'path_to_your_directory'
data_dir = os.path.join(os.path.dirname(__file__), 'data')
inputs_dir = os.path.join(data_dir, 'inputs')
outputs_dir = os.path.join(data_dir, 'outputs')

# input and output data names
input_data = 'Inputs_for_Test_Cases_Wotton2009.csv'
output_data = 'Outputs_for_Test_Cases_Wotton2009_cupy.csv'

# Create outputs dir
results_dir = os.path.join(outputs_dir, 'wotton2009')
os.makedirs(results_dir, exist_ok=True)

# Load CSV
data_path = os.path.join(inputs_dir, input_data) # path to input data
df = pd.read_csv(data_path, na_values=['NA', ''])

# Convert d0 and dj to nullable integers (preserve NaN values)
df['d0'] = df['d0'].astype('Int64')
df.loc[:, df.columns != 'd0'] = df.loc[:, df.columns != 'd0'].replace(np.nan, 0)

# Column order input
input_columns = [
    'fuel_type', 'wx_date', 'lat', 'long', 'elevation', 'slope', 'aspect',
    'ws', 'wd', 'ffmc', 'bui', 'pc', 'pdf', 'gfl', 'gcf', 'd0','dj'
]

# Output request: you can change this to ask for less outputs
output_request = ['tfc', 'hfros', 'hfi', 'cfb','fire_type', 'wsv', 'raz', 'be', 'sf', 'isi',
                  'fmc', 'sfc', 'rso', 'cfc', 'csfi']
outputs = []


#subset = df[df['id'].isin([6])]   # choose specific IDs
#for idx, row in subset.iterrows():
for idx, row in df.iterrows():
    try:
        # Convert fuel type  to numeric
        # ft_str = str(row['fuel_type']).strip().upper()
        # ft_num = fuel_type_lookup.get(ft_str)

        # Collect input values (it doesn't understand NA, leave empty)
        inputs = [None if (~np.isfinite(row[col]) | isinstance(row[col], type(pd.NA))) else row[col] for col in input_columns]
        inputs.append(output_request)

        # Initialize and run model
        fbp = FBP()
        fbp.initialize(*inputs)
        output = fbp.runFBP()

        # Store result with ID
        outputs.append(dict(zip(output_request, output), id=row['id']))

    except Exception as e:
        print(f'Error in row {idx} (ID {row.get("id")}): {e}')

# Save outputs
outputs_df = pd.DataFrame(outputs)
final_df = pd.merge(df, outputs_df, on='id', how='left')

final_df.to_csv(
    os.path.join(results_dir, output_data), # path for output data
    index=False
)


print('Done.')
