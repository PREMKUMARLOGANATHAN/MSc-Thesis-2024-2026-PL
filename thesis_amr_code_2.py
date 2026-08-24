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
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 2. User-defined Functions

import afn # Cao's FNN for choosing optimal embedding dimension
import CCM # Convergent Cross Mapping 
from ccm_result import ccm_result # Plot CCM result with bootstrapping
from ccm_result_1 import ccm_result_1 # Plot CCM results without bootstrapping
import extended_ccm # Delay CCM
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

# %% Other user defined functions

def has_consecutive_nans(series, threshold): # Thanks to GenAI
    
    nan_vals = series.isna()
    group = (~nan_vals).cumsum()
    streaks = nan_vals.groupby(group).cumsum()
    
    return (streaks >= threshold).any()

# %% AMR Data preprocessing

animals = ['Calves', 'Pigs', 'Poultry']
columns_of_interest = ['labIsolCode', 'matrix', 'sampY', 'sampM', 'Active Substance', 'MIC', 'cutoffValue']
amr_data = amr_df[columns_of_interest]

matrix_map = {'PRI 035': 'Pigs', 'PRI 036': 'Calves', 'PRI 019 Broilers': 'Poultry', 'PRI 019 Turkeys': 'Poultry'}
amr_data['Animal Type'] = amr_data['matrix'].map(matrix_map)
amr_data['Positive'] = (amr_data['MIC'] > amr_data['cutoffValue']).astype(int)
amr_data['YY-MM'] = amr_data['sampY'].astype(str) + '-' + amr_data['sampM'].astype(str)
amr_data['YY-MM'] = pd.to_datetime(amr_data['YY-MM'], format = '%Y-%m')
period_range = pd.date_range(start = amr_data['YY-MM'].min(),
                             end = amr_data['YY-MM'].max(), freq = 'MS')

gb_animal = ['YY-MM', 'Animal Type','labIsolCode', 'Active Substance', 'Positive']
gb_animal_table = amr_data[gb_animal]

gb_animal_res = gb_animal_table.groupby(['YY-MM', 'Active Substance', 'Animal Type'])['Positive'].agg(resistance = 'sum', tested = 'count').reset_index()
gb_animal_res['Resistance (%)'] = (gb_animal_res['resistance'] / gb_animal_res['tested']) * 100
gb_animal_res = gb_animal_res.sort_values(by = 'Active Substance')

# Raw Data

for i in gb_animal_res['Animal Type'].unique():
    data = gb_animal_res[gb_animal_res['Animal Type'] == i]
    fig, ax = plt.subplots(3, 5, sharex = True, figsize = (20, 12))
    ax = ax.flatten()
    
    for j, k in enumerate(data['Active Substance'].unique()):
        ab_data = data[data['Active Substance'] == k]
        ab_data.sort_values('YY-MM', inplace = True)
        ax[j].plot(ab_data['YY-MM'], ab_data['Resistance (%)'], 'o-', lw = 1.2, markersize = 2)
        ax[j].grid('--', c = 'grey', alpha = 0.3)
        ax[j].set_title(k, loc = 'left', size = 9)
        ax[j].tick_params('x', rotation = 90)
        
    fig.supxlabel('Time [Months]')
    fig.supylabel('Resistance (%)')
    fig.suptitle(f'Antimicrobial Resistance Pattern in {i} in Belgium')
    plt.tight_layout()
    plt.show()

animal_ab_table = pd.pivot_table(data = gb_animal_res, index = 'YY-MM', 
                                 columns = ['Animal Type', 'Active Substance'], values = 'Resistance (%)')
animal_ab_table = animal_ab_table.reindex(period_range)
animal_count_table = pd.pivot_table(data = gb_animal_res, index = 'YY-MM', 
                                 columns = ['Animal Type', 'Active Substance'], values = 'tested')
animal_count_table = animal_count_table.reindex(period_range)

# Raw Resistance (%) and No of isolates (n)

for i in sorted(gb_animal_res['Animal Type'].unique()):
    data = animal_ab_table[i]
    data_1 = animal_count_table[i]
    fig, ax = plt.subplots(4, 4, sharex = True, figsize = (20, 16))
    ax = ax.flatten()
    
    for j, k in enumerate(data.columns):
        ax[j].plot(data.index, data[k], 'o-', lw = 1.2, markersize = 2, color = 'blue')
        ax[j].grid('--', c = 'grey', alpha = 0.3)
        ax[j].set_title(k, size = 14, weight = 'bold')
        ax[j].tick_params('x', rotation = 90, labelsize = 14, labelbottom = True) if j > 10 else None 
        ax[j].tick_params('y', labelcolor = 'blue', labelsize = 14)
        
        ax_1 = ax[j].twinx()
        ax_1.plot(data_1.index, data_1[k], 'o-', c = 'red', lw = 1.2, markersize = 2, alpha = 0.3)
        ax_1.tick_params('y', labelcolor = 'red', labelsize = 14) 
        
    fig.supxlabel('Time [Months]', fontsize=15)
    fig.supylabel('Resistance (%)', color='blue', fontsize=15)
    
    fig.text(0.985, 0.50, 'No. of Isolates Tested (n)', va = 'center', 
        ha='center',  rotation = 90, color = 'red', fontsize = 15)
    fig.delaxes(ax[15])
    fig.suptitle(
        f'Antimicrobial Resistance Occurrence in {i} in Belgium', size=16, weight='bold')
    plt.tight_layout(rect=[0.005, 0.005, 0.975, 0.98])
    plt.savefig(f'AMR_levels_with_counts_in_{i}.png', dpi=600, bbox_inches='tight')

animals = sorted(gb_animal_res['Animal Type'].unique())

# Threshold selection

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
ax = ax.flatten()

for i, j in enumerate(animals):

    data = animal_count_table[j].iloc[:, 1]

    sns.histplot(data, bins='auto',kde=True,
        color='steelblue',edgecolor='black',linewidth=0.8,alpha=0.8,ax=ax[i])

    median = data.median()
    if j == 'Calves':
        q1 = 7
    else:
        q1 = int(data.quantile(0.25))
    ax[i].axvline(median, color='green', ls=':', lw=2,
                  label=f'Median = {median:.1f}')
    if j == 'Calves':
        ax[i].axvline(q1, color = 'purple', ls = '--', lw = 2,
            label = f'Threshold = {int(q1)}')
    else:
        ax[i].axvline(q1, color='purple', ls = '--', lw = 2,
                  label=f'Q1 = {int(q1)}')  
    ax[i].set_title(j, fontsize=13, weight='bold')
    ax[i].set_xlabel('Number of Resistant Isolates')
    ax[i].set_ylabel('Frequency')
    ax[i].legend(fontsize=9)

    mask_above = data.values >= q1
    mask_below = ~mask_above

    ax[i+3].scatter(
        data.index[mask_above],
        data.values[mask_above],
        c='steelblue',
        s=18,
        label='Above Threshold')

    ax[i+3].scatter(
        data.index[mask_below],
        data.values[mask_below],
        c='crimson',
        s=18,
        label='Below Threshold')

    ax[i+3].axhline(q1, color='purple', ls='--', lw=1.5)

    ax[i+3].grid(True, linestyle='--', alpha=0.4)
    ax[i+3].set_ylabel('No of Isolates Tested')
    ax[i+3].legend(fontsize=8)

fig.supxlabel('Time[Months]')
plt.tight_layout()
plt.savefig('no of isolates tested.png', dpi = 600)
plt.show()

##
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax = ax.flatten()

for i, j in enumerate(animals):

    data = animal_count_table[j].iloc[:, 1]

    median = data.median()
    if j == 'Calves':
        q1 = 7
    else:
        q1 = int(data.quantile(0.25))
    
    mask_above = data.values >= q1
    mask_below = ~mask_above

    ax[i].scatter(
        data.index[mask_above],
        data.values[mask_above],
        c='steelblue',
        s=18,
        label='Above Threshold')

    ax[i].scatter(
        data.index[mask_below],
        data.values[mask_below],
        c='crimson',
        s=18,
        label='Below Threshold')

    ax[i].axhline(q1, color='purple', ls='--', lw=1.5)

    ax[i].grid(True, linestyle='--', alpha=0.4)
    ax[i].set_ylabel('No of Isolates Tested', size = 15) if i == 0 else None
    ax[i].legend(fontsize=13.5)
    ax[i].set_title(j, fontsize = 14, weight = 'bold')
    ax[i].tick_params('x', rotation = 90)
    ax[i].tick_params('both', labelsize = 14)

fig.supxlabel('Time[Months]', size = 15)
plt.tight_layout()
plt.savefig('no of isolates tested.png', dpi = 600)
plt.show()

thresholds = {"Calves": 7, "Pigs": 12, "Poultry": 14}

filtered_animal_count_table = animal_count_table.copy()
filtered_animal_count_table['Calves'] = filtered_animal_count_table['Calves'] < thresholds['Calves']
filtered_animal_count_table['Pigs'] = filtered_animal_count_table['Pigs'] < thresholds['Pigs']
filtered_animal_count_table['Poultry'] = filtered_animal_count_table['Poultry'] < thresholds['Poultry']

filtered_animal_ab_table = animal_ab_table.copy()
filtered_animal_ab_table = filtered_animal_ab_table.mask(filtered_animal_count_table)

cols_to_drop = [col for col in filtered_animal_ab_table.columns if has_consecutive_nans(filtered_animal_ab_table[col], 5)]
animal_ab_df_cleaned = filtered_animal_ab_table.drop(columns = cols_to_drop)
loss = ((filtered_animal_ab_table.shape[1] - animal_ab_df_cleaned.shape[1]) / filtered_animal_ab_table.shape[1]) * 100
print(f'Loss: {np.round(loss, 2)}%')

# Now plot all again (3, 5 plot)

for i in gb_animal_res['Animal Type'].unique():
    data = filtered_animal_ab_table[i]
    data_1 = animal_count_table[i]
    fig, ax = plt.subplots(4, 4, sharex = True, figsize = (20, 16))
    ax = ax.flatten()
    
    for j, k in enumerate(data.columns):
        # ax[j].plot(data.index, data[k].interpolate(), 'o', lw = 1, markersize = 2.5, color = 'red', alpha = 0.8, label = 'imputed')
        ax[j].plot(data.index, data[k], 'o-', lw = 1.2, markersize = 3, color = 'blue', label = 'resistance')
        missing = data[k].isna()

        ax[j].plot(data.index[missing], [0] * missing.sum(), 'x', markersize=7, color='red', label='missing')
        ax[j].grid('--', c = 'grey', alpha = 0.3)
        ax[j].set_title(k, size = 13.5, weight = 'bold')
        ax[j].tick_params('x', rotation = 90, labelsize = 14, labelbottom = True) if j > 10 else None
        ax[j].legend(loc = 'upper right', fontsize = 13.5)
        ax[j].tick_params('y', labelsize = 14)
    
    fig.delaxes(ax[15])
    fig.supxlabel('Time [Months]', size = 15)
    fig.supylabel('Resistance (%)', size = 15)
    fig.suptitle(f'Antimicrobial Resistance Pattern in {i} in Belgium', weight = 'bold', size = 16)
    plt.tight_layout(rect = [0, 0, 1, 1])
    plt.savefig(f'AMR_occurance_with_gaps_and_how_filled_{i}.png', dpi = 600)
    plt.show()

# %% Climate Variable Smoothing

Temperature = preprocessing(temp.rolling(window = 5, center = True).mean().dropna())
Windspeed = preprocessing(ws.rolling(window = 3, center = True).mean().dropna())
RH = preprocessing(rh.rolling(window = 5, center = True).mean().dropna())
Precipitation = preprocessing(prec.rolling(window = 3, center = True).mean().dropna())

# %% Ampicillin and Chloramphenicol with temporal pattern

exclusion_map = {
    "Calves": ['Amikacin', 'Azithromycine', 'Cefotaxime', 'Ceftazidime',
                'Colistin', 'Gentamicin', 'Meropeneme', 'Nalidixic Acid', 'Tigecycline'],
    "Pigs": [
        "Amikacin", "Azithromycine", "Cefotaxime", "Ceftazidime",
        "Ciprofloxacin", "Colistin", "Gentamicin",
        "Meropeneme", "Nalidixic Acid", "Tigecycline"],
    "Poultry": [
        "Amikacin", "Azithromycine", "Cefotaxime", "Ceftazidime",
        "Colistin", "Gentamicin", "Meropeneme", "Tigecycline"]}

to_exclude = [(animal, abx) for animal, abx_list in exclusion_map.items() for abx in abx_list]

animal_ab_df_cleaned_new = animal_ab_df_cleaned.loc[:, ~animal_ab_df_cleaned.columns.isin(to_exclude)]
animal_ab_df_cleaned_filled = animal_ab_df_cleaned_new.interpolate()
animal_ab_df_cleaned_filled_smoothed = animal_ab_df_cleaned_filled.rolling(window = 5, center = True).mean()
animal_ab_df_cleaned_filled_smoothed = preprocessing(animal_ab_df_cleaned_filled_smoothed.dropna())

for animal in animals:
    ts = animal_ab_df_cleaned_filled[animal]
    yearly = ts.resample('YE').mean()
    yearly.index = yearly.index.year
    for i in yearly.columns:
        fig, ax = plt.subplots(figsize = (8, 5))
        ax.bar(yearly.index.astype(str), yearly[i].values, color = 'steelblue', edgecolor ='k', width = 0.7)
        
        ax.set_title(rf'Average {i}-resistant $\it{{E.\ coli}}$ occurance in {animal}', size = 15, weight = 'bold')
        ax.set_xlabel('Time [Year]', size = 15)
        ax.set_ylabel('Resistance (%)', size = 15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis = 'y', ls = '--', alpha = 0.4)
        ax.tick_params('both', labelsize = 14)
        plt.tight_layout()
        plt.savefig(f'{animal}_{i}_yearly_bar.png', dpi = 600, bbox_inches = True)
        plt.show()

for animal in animals:
        
    ts = animal_ab_df_cleaned_filled[animal]['Ampicillin']
    yearly = ts.resample('YE').mean()
    yearly.index = yearly.index.year
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        yearly.index.astype(str),
        yearly.values,
        color="steelblue",
        edgecolor="black",
        width=0.7)
    
    ax.set_title(
    rf"Average Ampicillin-resistant $\it{{E.\ coli}}$ occurrence in {animal} in Belgium",
    fontsize=16,
    weight="bold")
    ax.set_xlabel("Time [Year]", fontsize=15)
    ax.set_ylabel("Resistance (%)", fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.tick_params('both', labelsize = 14)
    
    plt.tight_layout()
    
for animal in animals:
        
    ts = animal_ab_df_cleaned_filled[animal]['Chloramphenicol']
    yearly = ts.resample('YE').mean()
    yearly.index = yearly.index.year
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        yearly.index.astype(str),
        yearly.values,
        color="steelblue",
        edgecolor="black",
        width=0.7)
    
    ax.set_title(
    rf"Average Chloramphenicol-resistant $\it{{E.\ coli}}$ occurrence in {animal} in Belgium",
    fontsize=16,
    weight="bold")
    ax.set_xlabel("Time [Year]", fontsize=15)
    ax.set_ylabel("Resistance (%)", fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.tick_params('both', labelsize = 14)
    plt.tight_layout()
    
    plt.savefig(f"AMR_chlor_{animal}.png", dpi=600)
    plt.show()

# Other ABs smoothed

fig, ax = plt.subplots(9, 2, figsize = (14, 24), sharex = True)
ax = ax.flatten()

for i, j in enumerate(animal_ab_df_cleaned_filled_smoothed.columns):
    
    ani, antibiotic = j
    ax[i].plot(animal_ab_df_cleaned_filled_smoothed.index, animal_ab_df_cleaned_filled_smoothed[j], lw = 1.2)
    ax[i].grid('--', alpha = 0.3, c = 'grey')
    ax[i].set_title(f'{ani} - {antibiotic}', size = 14, weight = 'bold')
    ax[i].tick_params('both', labelsize = 14)

fig.supxlabel('Time [Months]', size = 16)
fig.supylabel('Resistance (%)', size = 16)
fig.suptitle(r'Temporal pattern of $\it{E.\ coli}$ resistance', fontsize=16, weight = 'bold')
plt.tight_layout(rect = [0, 0.005, 0.993, 0.99])
plt.savefig('AMR_after_smoothing.png', dpi = 600)
plt.show()

# Ciprofloxacin and Nalidixic Acid in Poultry

poultry_abs = ['Ciprofloxacin', 'Nalidixic Acid']

fig, ax = plt.subplots(1, 2, figsize = (12, 6))
ax = ax.flatten()
for i, j in enumerate(poultry_abs):
    ax[i].plot(animal_ab_df_cleaned_filled.index, animal_ab_df_cleaned_filled.rolling(window = 5, center = True).mean()['Poultry'][j], lw = 1.2,)
    ax[i].grid('--', alpha = 0.3, c = 'grey')
    ax[i].set_title(f'Poultry - {j}', size = 15, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 15) if i == 0 else None
    ax[i].tick_params('x', rotation = 90)
    ax[i].tick_params('both', labelsize = 14)

fig.supxlabel('Time [Months]', size = 15)
plt.tight_layout()
plt.savefig('cipro_nali_poultry.png', dpi = 600, bbox_inches = 'tight')
plt.show()

# Sulfamethoxazole  and Trimethoprim in Pigs

pig_abs = ['Sulfamethoxazole', 'Trimethoprim']

fig, ax = plt.subplots(1, 2, figsize = (12, 6))
ax = ax.flatten()
for i, j in enumerate(pig_abs):
    ax[i].plot(animal_ab_df_cleaned_filled.index, animal_ab_df_cleaned_filled.rolling(window = 5, center = True).mean()['Pigs'][j], lw = 1.2,)
    ax[i].grid('--', alpha = 0.3, c = 'grey')
    ax[i].set_title(f'Poultry - {j}', size = 15, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 15) if i == 0 else None
    ax[i].tick_params('x', rotation = 90)
    ax[i].tick_params('both', labelsize = 14)

fig.supxlabel('Time [Months]', size = 15)
plt.tight_layout()
plt.savefig('trim_sulfa_pigs.png', dpi = 600, bbox_inches = 'tight')
plt.show()

# ABs with no temporal pattern

fig, ax = plt.subplots(7, 4, figsize = (16, 20), sharex = True)
ax = ax.flatten()

for i, j in enumerate(to_exclude):
    
    ani, antibiotic = j
    ax[i].plot(filtered_animal_ab_table.index, filtered_animal_ab_table[j])
    ax[i].set_title(f'{ani} - {antibiotic}', size = 13.5, weight = 'bold')
    ax[i].tick_params('y', labelsize = 14)
    ax[i].tick_params('x', labelsize = 14, labelbottom = True, rotation = 90) if i>=23 else None
    ax[i].set_ylim(-0.1, 1.1) if i == 6 or i == 16 or i == 19 or i == 25 else None
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f')) if i == 6 or i == 16 or i == 19 or i == 25 else None
    ax[i].grid('--', c = 'grey', alpha = 0.3)

fig.delaxes(ax[len(to_exclude)])
fig.supxlabel('Time [Months]', size = 15)
fig.supylabel('Resistance (%)', size = 15)
plt.tight_layout()
plt.savefig('amr_ab_with_no_pattern.png', dpi = 600)
plt.show()

for animal in animals:
    data = animal_ab_df_cleaned_filled[animal].mean(axis = 1)
    stl = STL(data.dropna(), period = 13)
    res = stl.fit()
    
    fig, ax = plt.subplots(4, 1, figsize = (8, 10), sharex= True)
    ax = ax.flatten()
    ax[0].plot(res.observed.index, res.observed, color = '#023047')
    ax[0].set_title('Observed', size = 14, loc = 'left', weight = 'bold')

    ax[1].plot(res.trend.index, res.trend, color = '#219ebc')
    ax[1].set_title(f'Trend', size = 14, loc = 'left', weight = 'bold')

    ax[2].plot(res.seasonal.index, res.seasonal, color = '#ffb703')
    ax[2].set_title(f'Seasonal Pattern', size = 14, loc = 'left', weight = 'bold')

    ax[3].plot(res.resid.index, res.resid, '.', color = '#fb8500' )
    ax[3].axhline(0, ls = '--', c = 'k', lw = 1.2)
    ax[3].set_title(f'Residual', size = 14, loc = 'left', weight = 'bold')
    ax[3].set_xticks(res.trend.index[4::6]) if animal == 'Calves' or animal == 'Pigs' else ax[3].set_xticks(res.trend.index[2::6])
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax[3].tick_params(axis='x', labelsize=14, rotation=90)
    # ax[3].tick_params('x', labelsize = 14, rotation = 90)
    [ax[i].tick_params('y', labelsize = 14) for i in range(len(ax))]
    [ax[i].grid('--', c='grey', alpha = 0.2) for i in range(len(ax))]

    fig.supxlabel('Time [Months]', size = 15)
    fig.supylabel('Resistance (%)', size = 15)
    fig.suptitle(f'Seasonal Decomposition of AMR in {animal}', size = 16, weight = 'bold')
    plt.tight_layout()
    plt.savefig(f'{animal}_AMR_trend.png', dpi = 600)
    plt.show()
    

# %% MDR profiles

iso_res = gb_animal_table.copy()
iso_res_animal = iso_res.groupby(['YY-MM', 'labIsolCode', 'Animal Type'])['Positive'].sum().reset_index()
iso_res_animal['Mono'] = iso_res_animal['Positive'] >= 1
iso_res_animal['Co-'] = iso_res_animal['Positive'] >= 2
iso_res_animal['MDR'] = iso_res_animal['Positive'] >=3

iso_res_animal_monthly = iso_res_animal.groupby(['YY-MM', 'Animal Type']).agg(
    n_isolates = ('labIsolCode', 'count'), Mono = ('Mono','sum'),
    CoR = ('Co-', 'sum'), MDR = ('MDR', 'sum')).reset_index()

iso_res_animal_monthly['Mono(%)'] = iso_res_animal_monthly['Mono'] / iso_res_animal_monthly['n_isolates']
iso_res_animal_monthly['CoR(%)'] = iso_res_animal_monthly['CoR'] / iso_res_animal_monthly['n_isolates']
iso_res_animal_monthly['MDR(%)'] = iso_res_animal_monthly['MDR'] / iso_res_animal_monthly['n_isolates']

thresholds = {"Calves": 7, "Pigs": 12, "Poultry": 14}
filtered_iso_res_animal_monthly = iso_res_animal_monthly.copy()
filtered_iso_res_animal_monthly['threshold'] = filtered_iso_res_animal_monthly['Animal Type'].map(thresholds)
low_sample_mask = filtered_iso_res_animal_monthly['n_isolates'] < filtered_iso_res_animal_monthly['threshold']
cols_to_null = ['Mono(%)', 'CoR(%)', 'MDR(%)']
filtered_iso_res_animal_monthly.loc[low_sample_mask, cols_to_null] = np.nan
filtered_iso_res_animal_monthly.drop(columns = ['threshold'], inplace = True)

# Raw

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (14, 5))
ax = ax.flatten()

for i, j in enumerate(sorted(filtered_iso_res_animal_monthly['Animal Type'].unique())):
    data = filtered_iso_res_animal_monthly[filtered_iso_res_animal_monthly['Animal Type'] == j]
    new_data = data.set_index('YY-MM').reindex(period_range)
    ax[i].plot(new_data.index, new_data['MDR(%)'] * 100, 'o-', lw = 2, c = 'steelblue', alpha = 0.9, markersize = 3)
    ax[i].grid('--', alpha = 0.4)
    ax[i].set_title(f'{j}', size = 14, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 15) if i == 0 else None
    ax[i].tick_params('x', rotation = 90, labelsize = 14)
    ax[i].tick_params('y', labelsize = 14)
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))

fig.suptitle('Temporal Trend of Multi Drug Resistant $\it{E.\ coli}$ in Food Producing Animals', size = 16, weight = 'bold')
fig.supxlabel('Time [Months]', size = 15)
plt.tight_layout()
plt.savefig('MDR_trend.png', dpi = 600)
# plt.savefig(fr'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\MDR_trend.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (14, 5))
ax = ax.flatten()

for i, j in enumerate(sorted(filtered_iso_res_animal_monthly['Animal Type'].unique())):
    data = filtered_iso_res_animal_monthly[filtered_iso_res_animal_monthly['Animal Type'] == j]
    new_data = data.set_index('YY-MM').reindex(period_range)
    ax[i].plot(new_data.index, new_data['Mono(%)'].interpolate() * 100, 'o-', lw = 2, c = 'steelblue', alpha = 0.9, markersize = 3)
    ax[i].grid('--', alpha = 0.4)
    ax[i].set_title(f'{j}', size = 14, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 15) if i == 0 else None
    ax[i].tick_params('x', rotation = 90, labelsize = 14)
    ax[i].tick_params('y', labelsize = 14)
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))

fig.suptitle(r'Temporal Trend of Mono Resistant $\it{E.\ coli}$ in Food Producing Animals', size = 16, weight = 'bold')
fig.supxlabel('Time [Months]', size = 15)
plt.tight_layout()
plt.savefig('Mono_trend.png', dpi = 600)
# plt.savefig(fr'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Mono_trend.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (14, 5))
ax = ax.flatten()

for i, j in enumerate(sorted(filtered_iso_res_animal_monthly['Animal Type'].unique())):
    data = filtered_iso_res_animal_monthly[filtered_iso_res_animal_monthly['Animal Type'] == j]
    new_data = data.set_index('YY-MM').reindex(period_range)
    ax[i].plot(new_data.index, new_data['CoR(%)'].interpolate() * 100, 'o-', lw = 2, c = 'steelblue', alpha = 0.9, markersize = 3)
    ax[i].grid('--', alpha = 0.4)
    ax[i].set_title(f'{j}', size = 14, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 15) if i == 0 else None
    ax[i].tick_params('x', rotation = 90, labelsize = 14)
    ax[i].tick_params('y', labelsize = 14)
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))

fig.suptitle('Temporal Trend of Co-Resistant $\it{E.\ coli}$ in Food Producing Animals', size = 16, weight = 'bold')
fig.supxlabel('Time [Months]', size = 15)
plt.tight_layout()
plt.savefig('CoR_trend.png', dpi = 600)
plt.savefig(fr'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CoR_trend.png', dpi = 600)
plt.show()

# After smoothing

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (14, 5))
ax = ax.flatten()

for i, j in enumerate(sorted(filtered_iso_res_animal_monthly['Animal Type'].unique())):
    data = filtered_iso_res_animal_monthly[filtered_iso_res_animal_monthly['Animal Type'] == j]
    new_data = data.set_index('YY-MM').reindex(period_range)
    ax[i].plot(new_data.index, new_data['MDR(%)'].interpolate().rolling(window = 5, center = True).mean() * 100, lw = 2, c = 'steelblue', alpha = 0.9, markersize = 3)
    ax[i].grid('--', alpha = 0.4)
    ax[i].set_title(f'{j}', size = 14, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 15) if i == 0 else None
    ax[i].tick_params('x', rotation = 90, labelsize = 14)
    ax[i].tick_params('y', labelsize = 14)
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))

fig.suptitle('Temporal Trend of Multi Drug Resistant $\it{E.\ coli}$ in Food Producing Animals', size = 16, weight = 'bold')
fig.supxlabel('Time [Months]', size = 15)
plt.tight_layout()
plt.savefig('MDR_trend_smoothed.png', dpi = 600)
plt.savefig(fr'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\MDR_trend.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (14, 5))
ax = ax.flatten()

for i, j in enumerate(sorted(filtered_iso_res_animal_monthly['Animal Type'].unique())):
    data = filtered_iso_res_animal_monthly[filtered_iso_res_animal_monthly['Animal Type'] == j]
    new_data = data.set_index('YY-MM').reindex(period_range)
    ax[i].plot(new_data.index, new_data['Mono(%)'].interpolate().interpolate().rolling(window = 5, center = True).mean() * 100, lw = 2, c = 'steelblue', alpha = 0.9, markersize = 3)
    ax[i].grid('--', alpha = 0.4)
    ax[i].set_title(f'{j}', size = 14, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 15) if i == 0 else None
    ax[i].tick_params('x', rotation = 90, labelsize = 14)
    ax[i].tick_params('y', labelsize = 14)
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))

fig.suptitle(r'Temporal Trend of Mono Resistant $\it{E.\ coli}$ in Food Producing Animals', size = 16, weight = 'bold')
fig.supxlabel('Time [Months]', size = 15)
plt.tight_layout()
plt.savefig('Mono_trend_smoothed.png', dpi = 600)
plt.savefig(fr'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Mono_trend.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (14, 5))
ax = ax.flatten()

for i, j in enumerate(sorted(filtered_iso_res_animal_monthly['Animal Type'].unique())):
    data = filtered_iso_res_animal_monthly[filtered_iso_res_animal_monthly['Animal Type'] == j]
    new_data = data.set_index('YY-MM').reindex(period_range)
    ax[i].plot(new_data.index, new_data['CoR(%)'].interpolate().rolling(window = 5, center = True).mean() * 100, lw = 2, c = 'steelblue', alpha = 0.9, markersize = 3)
    ax[i].grid('--', alpha = 0.4)
    ax[i].set_title(f'{j}', size = 14, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 15) if i == 0 else None
    ax[i].tick_params('x', rotation = 90, labelsize = 14)
    ax[i].tick_params('y', labelsize = 14)
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))

fig.suptitle('Temporal Trend of Co-Resistant $\it{E.\ coli}$ in Food Producing Animals', size = 16, weight = 'bold')
fig.supxlabel('Time [Months]', size = 15)
plt.tight_layout()
plt.savefig('CoR_trend_smoothed.png', dpi = 600)
plt.savefig(fr'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CoR_trend.png', dpi = 600)
plt.show()

mdr_data = filtered_iso_res_animal_monthly.copy()

mono_table = pd.pivot_table(data = mdr_data, columns = 'Animal Type', index = 'YY-MM', values = 'Mono(%)')
CoR_table = pd.pivot_table(data = mdr_data, columns = 'Animal Type', index = 'YY-MM', values = 'CoR(%)')
mdr_table = pd.pivot_table(data = mdr_data, columns = 'Animal Type', index = 'YY-MM', values = 'MDR(%)')

mono_table = mono_table.reindex(period_range).interpolate() * 100
CoR_table = CoR_table.reindex(period_range).interpolate() * 100
mdr_table = mdr_table.reindex(period_range).interpolate() * 100

mono_table_smoothed = preprocessing(mono_table.rolling(window = 5, center = True).mean().dropna())
CoR_table_smoothed = preprocessing(CoR_table.rolling(window = 5, center = True).mean().dropna())
mdr_table_smoothed = preprocessing(mdr_table.rolling(window = 5, center = True).mean().dropna())

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (14, 5))
ax = ax.flatten()

for i, j in enumerate(sorted(filtered_iso_res_animal_monthly['Animal Type'].unique())):
    data = filtered_iso_res_animal_monthly[filtered_iso_res_animal_monthly['Animal Type'] == j]
    new_data = data.set_index('YY-MM').reindex(period_range)
    ax[i].plot(new_data.index, new_data['MDR(%)'].interpolate() * 100, 'o-', lw = 2, c = 'steelblue', alpha = 0.9, markersize = 3, label = 'Original')
    # ax[i].plot(new_data.index[2:-2], mdr_table[j].rolling(window = 5, center = True).mean().dropna(), c = 'orange', lw = 2, alpha = 0.8, label = 'Smoothed')
    ax[i].grid('--', alpha = 0.4)
    ax[i].set_title(f'{j}', size = 14, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 15) if i == 0 else None
    ax[i].tick_params('x', rotation = 90, labelsize = 14)
    ax[i].tick_params('y', labelsize = 14)
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))
    ax[i].legend(fontsize = 13.5)

fig.suptitle('Temporal Trend of Multi Drug Resistant $\it{E.\ coli}$ in Food Producing Animals', size = 16, weight = 'bold')
fig.supxlabel('Time [Months]', size = 15)
plt.tight_layout()
plt.savefig(fr'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\MDR_trend.png', dpi = 600)
plt.show()

# %% CCM for Animal ABs

fig, ax = plt.subplots(5, 4, figsize = (15, 18), sharex = True)
ax = ax.flatten()

for i, j in enumerate(animal_ab_df_cleaned_filled_smoothed.columns):
    mutual_info = tdmi.tdmi(animal_ab_df_cleaned_filled_smoothed[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2, c = 'steelblue')
    ax[i].grid('--', alpha = 0.4)
    animal, antibiotic = j
    ax[i].set_title(f'{animal} - {antibiotic}' , fontsize=13.5, weight = 'bold')
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))
    ax[i].tick_params('x', labelbottom = True) if i > 13 else None
    ax[i].tick_params('both', labelsize = 14)
    
fig.delaxes(ax[18])
fig.delaxes(ax[19])
fig.suptitle('Average Mutual Information', size = 16, weight = 'bold')
fig.supxlabel('Time Delay (τ)', size = 15)
fig.supylabel('Mutual Information', size = 15)
plt.tight_layout(rect = [0, 0, 0.985, 0.985])
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Average_MI_18_combo.png', dpi = 600)
plt.show()

tau_all = [8, 5, 5, 8, 5, 6, 6, 8, 6, 7, 6, 6, 6, 6, 7, 5, 6, 6]
fig, ax = plt.subplots(5, 4, sharex = True, figsize = (15, 18))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(animal_ab_df_cleaned_filled_smoothed.columns):

    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(animal_ab_df_cleaned_filled_smoothed[j], e, tau_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
    animal, antibiotic = j
    E1 = [opt_E[k][0] / opt_E[k-1][0] for k in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(f'{animal} - {antibiotic}' , fontsize=13.5, weight = 'bold')
    ax[i].tick_params('x', labelbottom = True) if i > 13 else None
    ax[i].tick_params('both', labelsize = 14)

fig.delaxes(ax[18])
fig.delaxes(ax[19])
fig.supylabel('E1 Score', size = 15)
fig.supxlabel('No. of Embedding Dimension (D)', size = 15)
fig.suptitle("Cao's FNN for choosing optimal embedding dimension", size = 16, weight = 'bold')
plt.tight_layout(rect = [0, 0, 0.985, 0.985])
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Cao_FNN_18_AB.png', dpi = 600)
plt.show()

E_all = [4, 7, 6, 5, 7, 4, 6, 5, 6, 5, 4, 4, 5, 5, 5, 5, 4, 5]

fig, ax = plt.subplots(5, 4, subplot_kw = dict(projection = '3d'), figsize = (15, 18))
ax = ax.flatten()

for i, j in enumerate(animal_ab_df_cleaned_filled_smoothed.columns):
   
    M = mv.build_shadow(animal_ab_df_cleaned_filled_smoothed[j], E_all[i], tau_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    animal, antibiotic = j
    ax[i].set_title(f'{animal} - {antibiotic}' , fontsize=11, weight = 'bold')
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 14, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 14, pad = 0)
    ax[i].set_xlabel('AMR(t)', size = 15) if i == 0 else None
    ax[i].set_ylabel('AMR(t-τ)', size = 15) if i == 0 else None
    ax[i].set_zlabel('AMR(t-2τ)', size = 15) if i == 0 else None
fig.delaxes(ax[18])
fig.delaxes(ax[19])
fig.suptitle('Reconstructed Shadow Manifold of AMR levels in animals', size = 16, weight = 'bold')
plt.tight_layout(rect = [0, 0, 0.985, 0.985])
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\shadow_manifold_18_ab.png', dpi = 600)
plt.show()  

L_ranges = [np.arange(38, 88, 5), np.arange(43, 88, 5), np.arange(38, 88, 5,),
            np.arange(48, 88, 5), np.arange(43, 88, 5), np.arange(28, 88, 5,),
            np.arange(43, 88, 5), np.arange(48, 88, 5), np.arange(43, 88, 5,),
            np.arange(43, 88, 5), np.arange(28, 88, 5), np.arange(28, 88, 5,),
            np.arange(38, 88, 5), np.arange(38, 88, 5), np.arange(43, 88, 5,),
            np.arange(33, 88, 5), np.arange(28, 88, 5), np.arange(38, 88, 5,),]
titles = animal_ab_df_cleaned_filled_smoothed.columns

figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, animal_ab_df_cleaned_filled_smoothed.iloc[:,i].values, Temperature[2:].values.ravel(), L_ranges[i],
                          4, tau_all[i], 4, E_all[i], 'AMR', 'Temperature')
    animal, antibiotic = titles[i]
    fig_ax.set_title(f'{animal} - {antibiotic}', size = 13.5, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(5, 4, figsize=(19, 21))
ax = ax.flatten()

for i in range(18):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker())

    ax[i].set_title(source_ax.get_title(loc='center'), size = 13.5, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)
    ax[i].tick_params('both', labelsize = 14)
    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend(fontsize = 13.5)

fig.delaxes(ax[18])
fig.delaxes(ax[19])
fig.supxlabel('Time Series Length [L]', size = 15)
fig.supylabel(r'CCM Skill (ρ)', size = 15)
fig.suptitle('Convergent Cross Mapping of AMR and Temperature', size = 16, weight = 'bold')
plt.tight_layout(rect = [0, 0, 0.985, 0.985])
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Temp_AMR.png', dpi = 600)
plt.show()

figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, animal_ab_df_cleaned_filled_smoothed.iloc[:,i].values, RH[2:].values, L_ranges[i],
                          4, tau_all[i], 4, E_all[i], 'AMR', 'Relative Humidity')
    animal, antibiotic = titles[i]
    fig_ax.set_title(f'{animal} - {antibiotic}', size = 13.5, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(5, 4, figsize=(19, 21))
ax = ax.flatten()

for i in range(18):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker())

    ax[i].set_title(source_ax.get_title(loc='center'), size = 13.5, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)
    ax[i].tick_params('both', labelsize = 14)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend(fontsize = 12)

fig.delaxes(ax[18])
fig.delaxes(ax[19])
fig.supxlabel('Time Series Length [L]', size = 15)
fig.supylabel(r'CCM Skill (ρ)', size = 15)
fig.suptitle('Convergent Cross Mapping of AMR and Relative Humidity', size = 16, weight = 'bold')
plt.tight_layout(rect = [0, 0, 0.985, 0.985])
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_RH_AMR.png', dpi = 600)
plt.show()

figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, animal_ab_df_cleaned_filled_smoothed.iloc[:,i].values, Windspeed[3:-1].values, L_ranges[i],
                          4, tau_all[i], 4, E_all[i], 'AMR', 'Windspeed')
    animal, antibiotic = titles[i]
    fig_ax.set_title(f'{animal} - {antibiotic}', size = 13.5, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(5, 4, figsize=(19, 21))
ax = ax.flatten()

for i in range(18):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker())

    ax[i].set_title(source_ax.get_title(loc='center'), size = 13.5, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)
    ax[i].tick_params('both', labelsize = 14)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend(fontsize = 13.5)
    
fig.delaxes(ax[18])
fig.delaxes(ax[19])
fig.supxlabel('Time Series Length [L]', size = 15)
fig.supylabel(r'CCM Skill (ρ)', size = 15)
fig.suptitle('Convergent Cross Mapping of AMR and Windspeed', size = 16, weight = 'bold')
plt.tight_layout(rect = [0, 0, 0.985, 0.985])
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_WS_AMR.png', dpi = 600)
plt.show()

figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, animal_ab_df_cleaned_filled_smoothed.iloc[:,i].values, Precipitation[3:-1].values, L_ranges[i],
                          3, tau_all[i], 4, E_all[i], 'AMR', 'Precipitation')
    animal, antibiotic = titles[i]
    fig_ax.set_title(f'{animal} - {antibiotic}', size = 13.5, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(5, 4, figsize=(19, 21))
ax = ax.flatten()

for i in range(18):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker())

    ax[i].set_title(source_ax.get_title(loc='center'), size = 13.5, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)
    ax[i].tick_params('both', labelsize = 14)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend(fontsize = 13.5)

fig.delaxes(ax[18])
fig.delaxes(ax[19])       
fig.supxlabel('Time Series Length [L]', size = 15)
fig.supylabel(r'CCM Skill (ρ)', size = 15)
fig.suptitle('Convergent Cross Mapping of AMR and Precipitation', size = 16, weight = 'bold')
plt.tight_layout(rect = [0, 0, 0.985, 0.985])
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Prec_AMR.png', dpi = 600)
plt.show()

# %% CCM for MDR profiles

fig, ax = plt.subplots(1, 3, figsize = (12, 4), sharex = True)
ax = ax.flatten()

for i, j in enumerate(mono_table_smoothed.columns):
    mutual_info = tdmi.tdmi(mono_table_smoothed[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2, c = 'steelblue')
    ax[i].grid('--', alpha = 0.3)
    ax[i].set_title(j, size = 11, weight = 'bold')

fig.supxlabel('Time Delay (τ)')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information Plot - Mono-resistance', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Average_MI_Mono_MDR.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(1, 3, figsize = (12, 4), sharex = True)
ax = ax.flatten()

for i, j in enumerate(CoR_table_smoothed.columns):
    mutual_info = tdmi.tdmi(CoR_table_smoothed[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2, c = 'steelblue')
    ax[i].grid('--', alpha = 0.3)
    ax[i].set_title(j, size = 11, weight = 'bold')

fig.supxlabel('Time Delay (τ)')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information Plot - Co-resistance', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Average_MI_CoR_MDR.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(1, 3, figsize = (12, 4), sharex = True)
ax = ax.flatten()

for i, j in enumerate(mdr_table_smoothed.columns):
    mutual_info = tdmi.tdmi(mdr_table_smoothed[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2, c = 'steelblue')
    ax[i].grid('--', alpha = 0.3)
    ax[i].set_title(j, size = 11, weight = 'bold')

fig.supxlabel('Time Delay (τ)')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information Plot - Multi-Drug resistance', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Average_MI_MDR_MDR.png', dpi = 600)
plt.show()

tau_mono = [6, 5, 5]
tau_cor = [5, 4, 4]
tau_mdr = [5, 7, 6]

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (12, 4))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(mono_table_smoothed.columns):

    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(mono_table_smoothed[j], e, tau_mono[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
    E1 = [opt_E[k][0] / opt_E[k-1][0] for k in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11, weight = 'bold')

fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension (D)')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension - Mono-resistance", size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Cao_FNN_mono_resistance.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (12, 4))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(CoR_table_smoothed.columns):

    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(CoR_table_smoothed[j], e, tau_cor[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
   
    E1 = [opt_E[k][0] / opt_E[k-1][0] for k in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11, weight = 'bold')

fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension (D)')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension - Co-resistance", size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Cao_FNN_CoR_resistance.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (12, 4))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(mdr_table_smoothed.columns):

    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(mdr_table_smoothed[j], e, tau_mdr[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
    
    E1 = [opt_E[k][0] / opt_E[k-1][0] for k in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j , fontsize=11, weight = 'bold')

fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension (D)')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension - Multi-Drug resistance", size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Cao_FNN_mdr_resistance.png', dpi = 600)
plt.show()

E_mono = [5, 6, 5]
E_cor = [6, 5, 6]
E_mdr = [4, 6, 6]

L_range_mono = [np.arange(38, 88, 5), np.arange(38, 88, 5), np.arange(33, 88, 5)]
L_range_cor = [np.arange(38, 88, 5), np.arange(28, 88, 5), np.arange(28, 88, 5)]
L_range_mdr = [np.arange(28, 88, 5), np.arange(48, 88, 5), np.arange(43, 88, 5)]

fig, ax = plt.subplots(1, 3, subplot_kw = dict(projection = '3d'), figsize = (12, 4))
ax = ax.flatten()

for i, j in enumerate(mono_table.columns): 
    
    M = mv.build_shadow(mono_table[j], E_mono[i], tau_mono[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, fontsize=11, weight = 'bold')
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)
    ax[i].set_xlabel('AMR(t)', size = 9) if i == 0 else None
    ax[i].set_ylabel('AMR(t-τ)', size = 9) if i == 0 else None
    ax[i].set_zlabel('AMR(t-2τ)', size = 9) if i == 0 else None

fig.suptitle('Reconstructed Shadow Manifold of Mono - Resistance levels in animals', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\shadow_manifold_Mono_resistance.png', dpi = 600)
plt.show()   

fig, ax = plt.subplots(1, 3, subplot_kw = dict(projection = '3d'), figsize = (12, 4))
ax = ax.flatten()

for i, j in enumerate(CoR_table.columns): 
    
    M = mv.build_shadow(CoR_table[j], E_cor[i], tau_cor[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, fontsize=11, weight = 'bold')
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)
    ax[i].set_xlabel('AMR(t)', size = 9) if i == 0 else None
    ax[i].set_ylabel('AMR(t-τ)', size = 9) if i == 0 else None
    ax[i].set_zlabel('AMR(t-2τ)', size = 9) if i == 0 else None

fig.suptitle('Reconstructed Shadow Manifold of Co - Resistance levels in animals', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\shadow_manifold_Co_resistance.png', dpi = 600)
plt.show()   

fig, ax = plt.subplots(1, 3, subplot_kw = dict(projection = '3d'), figsize = (12, 4))
ax = ax.flatten()

for i, j in enumerate(mdr_table.columns): 
    
    M = mv.build_shadow(mdr_table[j], E_mdr[i], tau_mdr[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, fontsize=11, weight = 'bold')
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)
    ax[i].set_xlabel('AMR(t)', size = 9) if i == 0 else None
    ax[i].set_ylabel('AMR(t-τ)', size = 9) if i == 0 else None
    ax[i].set_zlabel('AMR(t-2τ)', size = 9) if i == 0 else None

fig.suptitle('Reconstructed Shadow Manifold of Multi - Drug Resistance levels in animals', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\shadow_manifold_mdr_resistance.png', dpi = 600)
plt.show()   

titles = mono_table.columns
# 1. Temperature

figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, mono_table_smoothed.iloc[:,i].values, Temperature[2:].values, L_range_mono[i], 
                          4, tau_mono[i], 4, E_mono[i], 'Mono-Res', 'Temperature')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(),
            color=line.get_color(), linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Mono-Resistance and Temperature', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Temp_Mono_Res.png', dpi = 600)
plt.show()

figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, mono_table_smoothed.iloc[:,i].values, Temperature[2:].values, L_range_mono[i], 
                          4, tau_mono[i], 4, E_mono[i], 'Mono-Res', 'Temperature')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Mono-Resistance and Temperature', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Temp_Mono_Res.png', dpi = 600)
plt.show()

figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, CoR_table_smoothed.iloc[:,i].values, Temperature[2:].values, L_range_cor[i], 
                          4, tau_cor[i], 4, E_cor[i], 'Co-Res', 'Temperature')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Co-Resistance and Temperature', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Temp_Co_Res.png', dpi = 600)
plt.show()

figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, mdr_table_smoothed.iloc[:,i].values, Temperature[2:].values, L_range_mdr[i], 
                          4, tau_mdr[i], 4, E_mdr[i], 'MDR', 'Temperature')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Multi-Drug Resistance and Temperature', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Temp_MDR_Res.png', dpi = 600)
plt.show()

# 2. Relative Humidity
figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, mono_table_smoothed.iloc[:,i].values, RH[2:].values, L_range_mono[i], 
                          4, tau_mono[i], 4, E_mono[i], 'Mono-Res', 'Humidity')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Mono-Resistance and Relative Humidity', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_RH_Mono_Res.png', dpi = 600)
plt.show()

figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, CoR_table_smoothed.iloc[:,i].values, RH[2:].values, L_range_cor[i], 
                          4, tau_cor[i], 4, E_cor[i], 'Co-Res', 'Humidity')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Co-Resistance and Relative Humidity', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_RH_Co_Res.png', dpi = 600)
plt.show()

figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, mdr_table_smoothed.iloc[:,i].values, RH[2:].values, L_range_mdr[i], 
                          4, tau_mdr[i], 4, E_mdr[i], 'MDR', 'Humidity')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Multi-Drug Resistance and Relative Humidity', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_RH_MDR_Res.png', dpi = 600)
plt.show()

# 3. Windspeed
figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, mono_table_smoothed.iloc[:,i].values, Windspeed[3:-1].values, L_range_mono[i], 
                          4, tau_mono[i], 4, E_mono[i], 'Mono-Res', 'Windspeed')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Mono-Resistance and Windspeed', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_WS_Mono_Res.png', dpi = 600)
plt.show()

figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, CoR_table_smoothed.iloc[:,i].values, Windspeed[3:-1].values, L_range_cor[i], 
                          4, tau_cor[i], 4, E_cor[i], 'Co-Res', 'Windspeed')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Co-Resistance and Windspeed', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_WS_Co_Res.png', dpi = 600)
plt.show()

figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, mdr_table_smoothed.iloc[:,i].values, Windspeed[3:-1].values, L_range_mdr[i], 
                          4, tau_mdr[i], 4, E_mdr[i], 'MDR', 'Windspeed')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Multi-Drug Resistance and Windspeed', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_WS_MDR_Res.png', dpi = 600)
plt.show()

# 4. Precipitation
figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, mono_table_smoothed.iloc[:,i].values, Precipitation[3:-1].values, L_range_mono[i], 
                          4, tau_mono[i], 4, E_mono[i], 'Mono-Res', 'Precipitation')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Mono-Resistance and Precipitation', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Prec_Mono_Res.png', dpi = 600)
plt.show()

figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, CoR_table_smoothed.iloc[:,i].values, Precipitation[3:-1].values, L_range_cor[i], 
                          4, tau_cor[i], 4, E_cor[i], 'Co-Res', 'Precipitation')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Co-Resistance and Precipitation', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Prec_Co_Res.png', dpi = 600)
plt.show()

figs = []
for i in range(3):
    fig_ax = ccm_result_1(CCM.CCM, mdr_table_smoothed.iloc[:,i].values, Precipitation[3:-1].values, L_range_mdr[i], 
                          4, tau_mdr[i], 4, E_mdr[i], 'MDR', 'Precipitation')
    
    fig_ax.set_title(titles[i], size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()

for i in range(3):

    source_ax = figs[i]

    for line in source_ax.lines:
        ax[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(),
            linestyle=line.get_linestyle(), linewidth=line.get_linewidth(), marker=line.get_marker())
    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:
        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of Multi-Drug Resistance and Precipitation', size = 13, weight = 'bold')
plt.tight_layout()
plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Prec_MDR_Res.png', dpi = 600)
plt.show()
