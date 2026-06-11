# %% Author Note

'''

This code is written by Prem Kumar Loganathan for the MSc Thesis titled, 
'Impact of Climate Change on AntiMicrobial Resistance'

All the class objects and user-defined functions needed for this code is made available in the same repository. 
No explanation will be provided anywhere in code until and unless necessary.

'''
# %% Importing Libraries

# 1. Pre-defined Functions

import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from python_calamine import CalamineWorkbook
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 2. User-defined Functions

import afn # Cao's FNN for choosing optimal embedding dimension
import CCM # Convergent Cross Mapping 
from ccm_result import ccm_result # Plot CCM result with bootstrapping
from ccm_result_1 import ccm_result_1 # Plot CCM results without bootstrapping
import manifold_visualisation as mv # Reconstruct shadow manifold
from preprocessing import preprocessing # Convert data to zero mean and unit variance
import tdmi # Fraser and Swinney Mutual Information method for choosing optimal time delay

# %% Loading Dataset

# 1) AMR Animals
amr_data_path = 'D:\Antibiotics_Usedata\EFSA data Ecoli.xlsx'
amr_df = pd.read_excel(amr_data_path, engine = 'calamine')

# 2) Temperature
temp_path = r'D:\MSc Thesis\Climate Data Files\BE_Temp_df.csv'
full_temp = pd.read_csv(temp_path)
temp = full_temp['Mean_Temp'][2:]

# 3) Relative Humidity
rh_path = r'D:\MSc Thesis\Climate Data Files\BE_Humi_df.csv'
rh = pd.read_csv(rh_path)
rh = rh['Mean_Hum'][2:]

# 4) Precipitation
prec_path = r'D:\MSc Thesis\Climate Data Files\BE_Prec_df.csv'
full_prec = pd.read_csv(prec_path)
prec = full_prec['Mean_Prec'][2:]

# 5) Windspeed
ws_path = r'D:\MSc Thesis\Climate Data Files\BE_Wind_df.csv'
full_ws = pd.read_csv(ws_path)
ws = full_ws['Mean_Wind'][2:]

# %% AMR Data preprocessing pt.1

columns_of_interest = ['labIsolCode', 'repCountry', 'matrix', 'sampY', 'sampM', 'sampD', 'Active Substance', 'MIC', 'cutoffValue']
amr_data = amr_df[columns_of_interest]

matrix_map = {'PRI 035': 'Pigs', 'PRI 036': 'Calves', 'PRI 019 Broilers': 'Poultry', 'PRI 019 Turkeys': 'Poultry'}
amr_data['Animal Type'] = amr_data['matrix'].map(matrix_map)
amr_data['Positive'] = (amr_data['MIC'] > amr_data['cutoffValue']).astype(int)
amr_data['YY-MM'] = amr_data['sampY'].astype(str) + '-' + amr_data['sampM'].astype(str)
amr_data['YY-MM'] = pd.to_datetime(amr_data['YY-MM'], format = '%Y-%m')
period_range = pd.date_range(start = amr_data['YY-MM'].min(),
                             end = amr_data['YY-MM'].max(), freq = 'MS')
counts = amr_data.groupby(['YY-MM', 'Animal Type', 'Active Substance'])[['labIsolCode']].nunique().reset_index()
resistant_counts = amr_data.groupby(['YY-MM', 'Animal Type', 'Active Substance'])[['Positive']].sum().reset_index()

full_data = resistant_counts.copy()
full_data = full_data.drop(columns = 'Positive')
full_data['Resistant'] = resistant_counts['Positive'] / counts['labIsolCode']

amr_table = pd.pivot_table(data = full_data, columns = ['Animal Type', 'Active Substance'], index = 'YY-MM', values = 'Resistant')
amr_table

'''
amr_data_grouped['Counts'] = [1] * len(amr_data_grouped)
amr_data_grouped_with_counts = amr_data_grouped.groupby(['YY-MM', 'Animal Type', 'Active Substance'])[['Counts', 'Positive']].sum().reset_index()

amr_data_pigs = amr_data_grouped_with_counts[amr_data_grouped_with_counts['Animal Type'] == 'Pigs']
amr_data_pigs_amp = amr_data_pigs[amr_data_pigs['Active Substance'] == 'Gentamicin']

amr_data_pigs_amp = (amr_data_pigs_amp.set_index('YY-MM').reindex(period_range).reset_index().rename(columns = {'index': 'YY-MM'}))

fig, ax = plt.subplots(figsize = (8, 6))
ax.plot(amr_data_pigs_amp['YY-MM'], amr_data_pigs_amp['Counts'], 'o-', color = 'grey', label = 'Tested')
ax.plot(amr_data_pigs_amp['YY-MM'], (amr_data_pigs_amp['Positive']/amr_data_pigs_amp['Counts']) * 100, 'o-', color = 'lightskyblue', label = 'Resistant')
ax.legend()
ax.set_xlabel('Time[Months]')
ax.set_ylabel('No of tested units \n& Resistance (%)')
ax.set_title('Ampicillin Resistant E. coli occurance in Pigs', loc = 'left')
plt.tight_layout()
plt.show()

amr_data_pigs_amp_q = (amr_data_pigs_amp.set_index('YY-MM')
    .resample('QE')[['Counts', 'Positive']].sum().reset_index())
fig, ax = plt.subplots(figsize = (8, 6))
ax.plot(amr_data_pigs_amp_q['YY-MM'], amr_data_pigs_amp_q['Counts'], 'o-', color = 'grey', label = 'Tested')
ax.plot(amr_data_pigs_amp_q['YY-MM'], (amr_data_pigs_amp_q['Positive']/amr_data_pigs_amp_q['Counts']) * 100, 'o-', color = 'lightskyblue', label = 'Resistant')
ax.legend()
ax.set_xlabel('Time[Quarters]')
ax.set_ylabel('No of tested units \n& Resistance (%)')
ax.set_title('Ampicillin Resistant E. coli occurance in Pigs', loc = 'left')
plt.tight_layout()
plt.show()

amr_data_pigs_amp_y = (amr_data_pigs_amp.set_index('YY-MM')
    .resample('YS')[['Counts', 'Positive']].sum().reset_index())
fig, ax = plt.subplots(figsize = (8, 6))
# ax.plot(amr_data_pigs_amp_y['YY-MM'], amr_data_pigs_amp_y['Counts'], 'o-', color = 'grey', label = 'Tested')
ax.plot(amr_data_pigs_amp_y['YY-MM'], (amr_data_pigs_amp_y['Positive']/amr_data_pigs_amp_y['Counts']) * 100, 'o-', color = 'lightskyblue', label = 'Resistant')
ax.legend()
ax.set_xlabel('Time[Years]')
ax.set_ylabel('No of tested units \n& Resistance (%)')
ax.set_title('Ampicillin Resistant E. coli occurance in Pigs', loc = 'left')
plt.tight_layout()
plt.show()

amr_data_grouped_with_counts['Resistance'] = (amr_data_grouped_with_counts['Positive'] / amr_data_grouped_with_counts['Counts']) * 100

table_for_grouped = pd.pivot_table(data = amr_data_grouped_with_counts, index = 'YY-MM', columns = ['Animal Type', 'Active Substance'], values = 'Resistance')
table_for_grouped = (table_for_grouped.reindex(period_range).reset_index().rename(columns = {'index': 'YY-MM'}))
'''
def has_consecutive_nans(series, threshold): # Thanks to GenAI
    
    nan_vals = series.isna()
    group = (~nan_vals).cumsum()
    streaks = nan_vals.groupby(group).cumsum()
    
    return (streaks >= threshold).any()

cols_to_drop = [col for col in res_table.columns if has_consecutive_nans(res_table[col], 5)]
amr_df_cleaned = res_table.drop(columns = cols_to_drop)
loss = ((res_table.shape[1] - amr_df_cleaned.shape[1]) / res_table.shape[1]) * 100
print(f'Loss: {np.round(loss, 2)}%')

# amr_df_cleaned = amr_df_cleaned.set_index('YY-MM')
# Plotting the figure to see if the inconsistency is continous

fig, ax = plt.subplots(6, 7, figsize=(16, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(amr_df_cleaned.columns):
    ax[i].plot(amr_df_cleaned.index, amr_df_cleaned[j], 'o-', label = j, markersize = 2)
    ax[i].set_title(j, loc = 'left', size = 8)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_xticks(amr_df_cleaned.index[::12])
    ax[i].tick_params('x', rotation = 90)

ABs_with_100_and_0_resistance = ['Cefotaxime', 'Meropeneme', 'Ceftazidime', 'Colistin', 'Tigecycline'] 
amr_df_cleaned_2 = amr_df_cleaned.loc[:, ~amr_df_cleaned.columns.get_level_values(1).isin(ABs_with_100_and_0_resistance)]