################# Process data and get a new dataset ######################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle

# Create a function to process data
def process(data, feature):
    # Convert 'date' column to pandas datetime object
    data['date'] = pd.to_datetime(data['date'])

    # Set 'date' column as the index and remove any duplicate indices
    data = data.set_index('date')
    data = data[~data.index.duplicated()]

    # Create a new index with a minute frequency from 1/1/2020 to 1/1/2021
    new_index = pd.date_range(start='1/1/2020', end='1/1/2021', freq='1T').rename('date')

    # Reindex the DataFrame using the new index and convert the timezone of the 'date' column
    data = data.reindex(new_index)
    data.reset_index(inplace=True)
    data['date'] = data['date'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
    data = data.set_index('date')

    # Split the data into two sections based on the date and concatenate them in a different order
    data1 = data.loc[(data.index < '2020-03-08')]
    data2 = data.loc[(data.index >= '2020-03-09')]
    data_ = pd.concat([data1, data2], ignore_index=False)

    # Split the concatenated data again based on the date and reset the index
    data3 = data_.loc[(data_.index < '2020-11-01')]
    data4 = data_.loc[(data_.index >= '2020-11-02')]
    data = pd.concat([data3, data4], ignore_index=False)
    data.reset_index(inplace=True)

    # Convert the 'date' column to a string format of 'YYYY-MM-DDTHH:MM:SS.000Z'
    data['date'] = data['date'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    # Rename the 'date' column to 'timestamp' and set it as the index
    data = data.rename(columns={'date': 'timestamp'})
    data = data.set_index('timestamp')

    # Calculate the mean of each row and add a 'value' column with the result
    data['value'] = data.mean(axis=1)

    # Convert the 'value' column to a different unit based on the feature parameter
    if feature == 't':
        data['value'] = data['value'].apply(lambda x: ((x - 32) * 5 / 9))
    elif feature == 'm':
        data['value'] = data['value'].apply(lambda x: 1.699 * x / 3600)
    elif feature == 'm_gpm':
        data['value'] = data['value'].apply(lambda x: 1.699 * 0.13368 * x / 3600)
    else:
        pass

    # Keep only the 'series' and 'value' columns
    data = data[['series', 'value']]

    # Return the cleaned DataFrame
    return data

path = "C:/Users/Antonio/Desktop/Projects/PIANN_Singapore_ACM/Data/lbnlbldg59/"

# Define thermal properties
cp_a = 1.006
cp_w = 4.200
rho_a = 1.204
rho_w = 1000

# Get and process averate outdoor air
rtu_oa_t_avg = pd.read_csv(path+"rtu_oa_t.csv")
rtu_oa_t_avg['series'] = 't_oa_avg [°C]'
rtu_oa_t_avg = process(rtu_oa_t_avg, 't')

# Get and process averate indoor air
rtu_ra_t_avg = pd.read_csv(path+"rtu_ra_t.csv")
rtu_ra_t_avg['series'] = 't_ra_avg [°C]'
rtu_ra_t_avg = process(rtu_ra_t_avg, 't')

# Get and process HP variables
t_hwr = pd.read_csv(path+"ashp_hw.csv")[['date','aru_001_hwr_temp']]
t_hwr['series'] = 't_hwr [°C]'
t_hwr = process(t_hwr, 't')

m_hws = pd.read_csv(path+"ashp_hw.csv")[['date','aru_001_hws_fr_gpm']]
m_hws['series'] = 'm_hws [m3/s]'
m_hws = process(m_hws, 'm_gpm')

t_hws = pd.read_csv(path+"ashp_hw.csv")[['date','aru_001_hws_temp']]
t_hws['series'] = 't_hws [°C]'
t_hws = process(t_hws, 't')

# Initialize cooling flow rate
Q_cw = rtu_ra_t_avg.copy()
Q_cw['series'] = 'Q_cool [kW]'
Q_cw['value'] = 0

# Initialize statistics variables
max_rtu_ma_t = []
min_rtu_ma_t = []
max_rtu_sa_t = []
min_rtu_sa_t = []
max_rtu_sa_fr = []
min_rtu_sa_fr = []
max_rtu_oa_t = []
min_rtu_oa_t = []
max_rtu_ra_t = []
min_rtu_ra_t = []

# Iterate over each RTU
for i in range(1,5):

    # Get and process variables from each RTU
    rtu_ma_t = pd.read_csv(path+"rtu_ma_t.csv")[['date','rtu_00'+str(i)+'_ma_temp']]
    rtu_ma_t['series'] = 't_ma [°C]'
    rtu_ma_t = process(rtu_ma_t, 't')

    rtu_sa_t = pd.read_csv(path+"rtu_sa_t.csv")[['date','rtu_00'+str(i)+'_sa_temp']]
    rtu_sa_t['series'] = 't_sa [°C]'
    rtu_sa_t = process(rtu_sa_t, 't')

    m_sa = pd.read_csv(path+"rtu_sa_fr.csv")[['date','rtu_00'+str(i)+'_fltrd_sa_flow_tn']]
    m_sa['series'] = 'm_sa [m3/s]'
    m_sa = process(m_sa, 'm')

    rtu_oa_t = pd.read_csv(path+"rtu_oa_t.csv")[['date','rtu_00'+str(i)+'_oa_temp']]
    rtu_oa_t['series'] = 't_oa [°C]'
    rtu_oa_t = process(rtu_oa_t, 't')

    rtu_ra_t = pd.read_csv(path+"rtu_ra_t.csv")[['date','rtu_00'+str(i)+'_ra_temp']]
    rtu_ra_t['series'] = 't_ra [°C]'
    rtu_ra_t = process(rtu_ra_t, 't')

    # Sum the cooling flow rate given by each RTU
    Q_cw['value'] = Q_cw['value'] + m_sa['value'] * cp_a * rho_a * (rtu_ma_t['value'] - rtu_sa_t['value'])

    # Get statistics
    max_rtu_ma_t = np.append(max_rtu_ma_t, np.nanmax(rtu_ma_t['value'].values))
    min_rtu_ma_t = np.append(min_rtu_ma_t, np.nanmin(rtu_ma_t['value'].values))
    max_rtu_sa_t = np.append(max_rtu_sa_t, np.nanmax(rtu_sa_t['value'].values))
    min_rtu_sa_t = np.append(min_rtu_sa_t, np.nanmin(rtu_sa_t['value'].values))
    max_rtu_sa_fr = np.append(max_rtu_sa_fr, np.nanmax(m_sa['value'].values))
    min_rtu_sa_fr = np.append(min_rtu_sa_fr, np.nanmin(m_sa['value'].values))
    max_rtu_oa_t = np.append(max_rtu_oa_t, np.nanmax(rtu_oa_t['value'].values))
    min_rtu_oa_t = np.append(min_rtu_oa_t, np.nanmin(rtu_oa_t['value'].values))
    max_rtu_ra_t = np.append(max_rtu_ra_t, np.nanmax(rtu_ra_t['value'].values))
    min_rtu_ra_t = np.append(min_rtu_ra_t, np.nanmin(rtu_ra_t['value'].values))

Q_cool = Q_cw.copy()

# Initialize heating flow rate
Q_hw = m_hws.copy()
Q_hw['series'] = 'Q_heat [kW]'

# Compute heating flow rate
Q_hw['value'] = m_hws['value']*cp_w*rho_w*(t_hws['value']-t_hwr['value'])

Q_heat = Q_hw.copy()

# Create a dataset based on the processed variables
dataset = pd.concat([rtu_oa_t_avg, rtu_ra_t_avg, Q_cool, Q_heat],ignore_index=False)

# Sort by timestamp and index
dataset = dataset.sort_values(by = ['timestamp', 'series'])
dataset.reset_index(inplace=True)

# Add and empty column label (in order to upload to TRAINSET)  and rearrange the columns
dataset['label'] = ''
dataset = dataset[['series', 'timestamp', 'value', 'label']]

# Take not nan rows
dataset = dataset[dataset['value'].notna()]

# Convert timestamp column to datetime and set as index
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
dataset.set_index('timestamp', inplace=True)

# Create indeces for the columns where the mean operation should be used for resampling
columns_mean = ['Q_cool [kW]', 'Q_heat [kW]','t_oa_avg [°C]']
columns_exact = ['t_ra_avg [°C]']

# Group rows by series
grouped_data = dataset.groupby('series')

# Resample data by mean
resampled_data = grouped_data.apply(lambda df: df.resample('30T').agg([(col, 'mean') for col in columns_mean] + [(col, 'last') for col in columns_exact]))
resampled_data = resampled_data.reset_index()
resampled_data.columns = resampled_data.columns.droplevel(0)

# Adjust columns order
resampled_data.columns = ['series', 'timestamp', 'Q_cool [kW]', 'Q_heat [kW]','t_ra_avg [°C]', 't_oa_avg [°C]']

# Add empty columns value and label (in order to upload to TRAINSET)
resampled_data['value'] = ''
resampled_data['label'] = ''
resampled_data["value"] = resampled_data.apply(lambda x: x[x["series"]], axis=1)
resampled_data.drop(['Q_cool [kW]', 'Q_heat [kW]','t_ra_avg [°C]', 't_oa_avg [°C]'], axis=1, inplace=True)

# Convert timestamp to string
resampled_data['timestamp'] = resampled_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

# Sort by timestamp and series and reset index
resampled_data = resampled_data.sort_values(by = ['timestamp', 'series'])
resampled_data.reset_index(inplace=True,drop=True)

# Select periods, set index to series and take not nan values
resampled_data = resampled_data.iloc[16*4:-(33*4-1)]
resampled_data = resampled_data.set_index('series')
resampled_data = resampled_data[resampled_data['value'].notna()]

# Save the processed dataset as csv
resampled_data.to_csv(path+'processed/dataset_new.csv')

# Print statistics
print("old dataset")
print("t_sa max", np.nanmax(max_rtu_sa_t))
print("t_sa min", np.nanmin(min_rtu_sa_t))
print("t_ra max", np.nanmax(max_rtu_ra_t))
print("t_ra min", np.nanmin(min_rtu_ra_t))
print("t_ma max", np.nanmax(max_rtu_ma_t))
print("t_ma min", np.nanmin(min_rtu_ma_t))
print("t_oa max", np.nanmax(max_rtu_oa_t))
print("t_oa min", np.nanmin(min_rtu_oa_t))
print("m_sa max", np.nanmax(max_rtu_sa_fr))
print("m_sa min", np.nanmin(min_rtu_sa_fr))
print("t_hwr max", np.nanmax(t_hwr['value'].values))
print("t_hwr min", np.nanmin(t_hwr['value'].values))
print("t_hws max", np.nanmax(t_hws['value'].values))
print("t_hws min", np.nanmin(t_hws['value'].values))
print("m_hws max", np.nanmax(m_hws['value'].values))
print("m_hws max", np.nanmin(m_hws['value'].values))
print("new dataset")
print("Q_cool max", np.max(resampled_data[resampled_data.index == 'Q_cool [kW]']['value'].values))
print("Q_cool min", np.min(resampled_data[resampled_data.index == 'Q_cool [kW]']['value'].values))
print("Q_heat max", np.max(resampled_data[resampled_data.index == 'Q_heat [kW]']['value'].values))
print("Q_heat min", np.min(resampled_data[resampled_data.index == 'Q_heat [kW]']['value'].values))
print("t_ra_avg max", np.max(resampled_data[resampled_data.index == 't_ra_avg [°C]']['value'].values))
print("t_ra_avg min", np.min(resampled_data[resampled_data.index == 't_ra_avg [°C]']['value'].values))
print("t_oa_avg max", np.max(resampled_data[resampled_data.index == 't_oa_avg [°C]']['value'].values))
print("t_oa_avg min", np.min(resampled_data[resampled_data.index == 't_oa_avg [°C]']['value'].values))






