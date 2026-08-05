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

for i in gb_animal_res['Animal Type'].unique():
    data = gb_animal_res[gb_animal_res['Animal Type'] == i]
    fig, ax = plt.subplots(3, 5, sharex = True, figsize = (20, 12))
    ax = ax.flatten()
    
    for j, k in enumerate(data['Active Substance'].unique()):
        ab_data = data[data['Active Substance'] == k]
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

for i in sorted(gb_animal_res['Animal Type'].unique()):
    data = animal_ab_table[i]
    data_1 = animal_count_table[i]
    fig, ax = plt.subplots(3, 5, sharex = True, figsize = (20, 12))
    ax = ax.flatten()
    
    for j, k in enumerate(data.columns):
        ax[j].plot(data.index, data[k], 'o-', lw = 1.2, markersize = 2, color = 'blue')
        ax[j].grid('--', c = 'grey', alpha = 0.3)
        ax[j].set_title(k, size = 11)
        ax[j].tick_params('x', rotation = 90)
        ax[j].tick_params('y', labelcolor = 'blue')
        
        ax_1 = ax[j].twinx()
        ax_1.plot(data_1.index, data_1[k], 'o-', c = 'red', lw = 1.2, markersize = 2, alpha = 0.3)
        ax_1.tick_params('y', labelcolor = 'red') 
        
    fig.supxlabel('Time [Months]', fontsize=13)
    fig.supylabel('Resistance (%)', color='blue', fontsize=13)
    
    fig.text(
        0.985, 0.50,
        'No. of Isolates Tested (n)',
        va='center',
        ha='center',
        rotation=-90,
        color='red',
        fontsize=13
    )
    
    fig.suptitle(
        f'Antimicrobial Resistance Occurrence in {i} in Belgium',
        fontsize=15,
        fontweight='bold'
    )
    
    plt.tight_layout(rect=[0.04, 0.04, 0.97, 0.95])
    
    plt.savefig(
        f'AMR_levels_with_counts_in_{i}.png',
        dpi=600,
        bbox_inches='tight'
    )

sns.set_theme(style="whitegrid", font_scale=1.1)

animals = sorted(gb_animal_res['Animal Type'].unique())

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
    # ax[i+3].set_xlabel('Index')
    ax[i+3].set_ylabel('No of Isolates Tested')
    ax[i+3].legend(fontsize=8)

fig.supxlabel('Time[Months]')
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
    fig, ax = plt.subplots(3, 5, sharex = True, figsize = (20, 12))
    ax = ax.flatten()
    
    for j, k in enumerate(data.columns):
        ax[j].plot(data.index, data[k].interpolate(), 'o', lw = 1, markersize = 2.5, color = 'red', alpha = 0.8, label = 'imputed')
        ax[j].plot(data.index, data[k], 'o-', lw = 1.2, markersize = 3, color = 'blue', label = 'resistance')
        ax[j].grid('--', c = 'grey', alpha = 0.3)
        ax[j].set_title(k)
        ax[j].tick_params('x', rotation = 90)
        ax[j].legend(loc = 'upper right')
        
    fig.supxlabel('Time [Months]')
    fig.supylabel('Resistance (%)')
    fig.suptitle(f'Antimicrobial Resistance Pattern in {i} in Belgium', weight = 'bold')
    plt.tight_layout()
    plt.savefig(f'AMR_occurance_with_gaps_and_how_filled_{i}.png', dpi = 600)
    plt.show()

# %% Climate Variable Smoothing

Temperature = preprocessing(temp.rolling(window = 5, center = True).mean().dropna())
Windspeed = preprocessing(ws.rolling(window = 3, center = True).mean().dropna())
RH = preprocessing(rh.rolling(window = 5, center = True).mean().dropna())
Precipitation = preprocessing(prec.rolling(window = 3, center = True).mean().dropna())

# %% AB with temporal pattern
# %% AB with temporal pattern

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
    fontsize=12,
    weight="bold")
    ax.set_xlabel("Time [Year]", fontsize=12)
    ax.set_ylabel("Resistance (%)", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
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
    fontsize=12,
    weight="bold")
    ax.set_xlabel("Time [Year]", fontsize=12)
    ax.set_ylabel("Resistance (%)", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    
    plt.savefig(f"AMR_chlor_{animal}.png", dpi=600)
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

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (12, 4))
ax = ax.flatten()

for i, j in enumerate(sorted(filtered_iso_res_animal_monthly['Animal Type'].unique())):
    data = filtered_iso_res_animal_monthly[filtered_iso_res_animal_monthly['Animal Type'] == j]
    new_data = data.set_index('YY-MM').reindex(period_range)
    ax[i].plot(new_data.index, new_data['MDR(%)'].interpolate() * 100, 'o-', lw = 2, c = 'steelblue', alpha = 0.9, markersize = 3)
    ax[i].grid('--', alpha = 0.4)
    ax[i].set_title(f'{j}', size = 12, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 11) if i == 0 else None
    ax[i].tick_params('x', rotation = 90, labelsize = 9)
    ax[i].tick_params('y', labelsize = 9)
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))

fig.suptitle('Temporal Trend of Multi Drug Resistant $\it{E.\ coli}$ in Food Producing Animals', size = 13, weight = 'bold')
fig.supxlabel('Time [Months]', size = 11)
plt.tight_layout()
plt.savefig('MDR_trend.png', dpi = 600)
# plt.savefig(fr'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\MDR_trend.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (12, 4))
ax = ax.flatten()

for i, j in enumerate(sorted(filtered_iso_res_animal_monthly['Animal Type'].unique())):
    data = filtered_iso_res_animal_monthly[filtered_iso_res_animal_monthly['Animal Type'] == j]
    new_data = data.set_index('YY-MM').reindex(period_range)
    ax[i].plot(new_data.index, new_data['Mono(%)'].interpolate() * 100, 'o-', lw = 2, c = 'steelblue', alpha = 0.9, markersize = 3)
    ax[i].grid('--', alpha = 0.4)
    ax[i].set_title(f'{j}', size = 12, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 11) if i == 0 else None
    ax[i].tick_params('x', rotation = 90, labelsize = 9)
    ax[i].tick_params('y', labelsize = 9)
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))

fig.suptitle(r'Temporal Trend of Mono Resistant $\it{E.\ coli}$ in Food Producing Animals', size = 13, weight = 'bold')
fig.supxlabel('Time [Months]', size = 11)
plt.tight_layout()
plt.savefig('Mono_trend.png', dpi = 600)
# plt.savefig(fr'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Mono_trend.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(1, 3, sharex = True, figsize = (12, 4))
ax = ax.flatten()

for i, j in enumerate(sorted(filtered_iso_res_animal_monthly['Animal Type'].unique())):
    data = filtered_iso_res_animal_monthly[filtered_iso_res_animal_monthly['Animal Type'] == j]
    new_data = data.set_index('YY-MM').reindex(period_range)
    ax[i].plot(new_data.index, new_data['CoR(%)'].interpolate() * 100, 'o-', lw = 2, c = 'steelblue', alpha = 0.9, markersize = 3)
    ax[i].grid('--', alpha = 0.4)
    ax[i].set_title(f'{j}', size = 12, weight = 'bold')
    ax[i].set_ylabel('Resistance (%)', size = 11) if i == 0 else None
    ax[i].tick_params('x', rotation = 90, labelsize = 9)
    ax[i].tick_params('y', labelsize = 9)
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))

fig.suptitle('Temporal Trend of Co-Resistant $\it{E.\ coli}$ in Food Producing Animals', size = 13, weight = 'bold')
fig.supxlabel('Time [Months]', size = 11)
plt.tight_layout()
plt.savefig('CoR_trend.png', dpi = 600)
# plt.savefig(fr'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CoR_trend.png', dpi = 600)
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
# %% CCM for Animal ABs

fig, ax = plt.subplots(3, 6, figsize = (24, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(animal_ab_df_cleaned_filled_smoothed.columns):
    mutual_info = tdmi.tdmi(animal_ab_df_cleaned_filled_smoothed[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2, c = 'steelblue')
    ax[i].grid('--', alpha = 0.4)
    # ax[i].set_xlabel('Time Delay (τ)') if i == 13 else None
    # ax[i].set_ylabel('Mutual Information') if i == 6 else None
    animal, antibiotic = j
    ax[i].set_title(f'{animal} - {antibiotic}' , fontsize=11, weight = 'bold')
    ax[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}"))

fig.suptitle('Average Mutual Information', size = 13, weight = 'bold')
fig.supxlabel('Time Delay (τ)')
fig.supylabel('Mutual Information')
plt.tight_layout()
# plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Average_MI_18_combo.png', dpi = 600)
plt.show()

tau_all = [8, 5, 5, 8, 5, 6, 6, 8, 6, 7, 6, 6, 6, 6, 7, 5, 6, 6]
fig, ax = plt.subplots(3, 6, sharex = True, figsize = (24, 12))
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
    ax[i].set_title(f'{animal} - {antibiotic}' , fontsize=11, weight = 'bold')

fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension (D)')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension", size = 13, weight = 'bold')
plt.tight_layout()
# plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Cao_FNN_18_AB.png', dpi = 600)
plt.show()

E_all = [4, 7, 6, 5, 7, 4, 6, 5, 6, 5, 4, 4, 5, 5, 5, 5, 4, 5]

fig, ax = plt.subplots(3, 6, subplot_kw = dict(projection = '3d'), figsize = (24, 12))
ax = ax.flatten()

for i, j in enumerate(animal_ab_df_cleaned_filled_smoothed.columns):
   
    M = mv.build_shadow(animal_ab_df_cleaned_filled_smoothed[j], E_all[i], tau_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    animal, antibiotic = j
    ax[i].set_title(f'{animal} - {antibiotic}' , fontsize=11, weight = 'bold')
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)
    ax[i].set_xlabel('AMR(t)', size = 9) if i == 0 else None
    ax[i].set_ylabel('AMR(t-τ)', size = 9) if i == 0 else None
    ax[i].set_zlabel('AMR(t-2τ)', size = 9) if i == 0 else None

fig.suptitle('Reconstructed Shadow Manifold of AMR levels in animals', size = 13, weight = 'bold')
plt.tight_layout()
# plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\shadow_manifold_18_ab.png', dpi = 600)
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
    fig_ax = ccm_result_1(CCM.CCM, animal_ab_df_cleaned_filled_smoothed.iloc[:,i].values, Temperature[2:].values, L_ranges[i],
                          4, tau_all[i], 4, E_all[i], 'AMR', 'Temperature')
    animal, antibiotic = titles[i]
    fig_ax.set_title(f'{animal} - {antibiotic}', size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(3, 6, figsize=(24, 12))
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

    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
       
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of AMR and Temperature', size = 13, weight = 'bold')
plt.tight_layout()
# plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Temp_AMR.png', dpi = 600)
plt.show()

figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, animal_ab_df_cleaned_filled_smoothed.iloc[:,i].values, RH[2:].values, L_ranges[i],
                          4, tau_all[i], 4, E_all[i], 'AMR', 'Relative Humidity')
    animal, antibiotic = titles[i]
    fig_ax.set_title(f'{animal} - {antibiotic}', size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(3, 6, figsize=(24, 12))
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

    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
       
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of AMR and Relative Humidity', size = 13, weight = 'bold')
plt.tight_layout()
# plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_RH_AMR.png', dpi = 600)
plt.show()

figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, animal_ab_df_cleaned_filled_smoothed.iloc[:,i].values, Windspeed[3:-1].values, L_ranges[i],
                          4, tau_all[i], 4, E_all[i], 'AMR', 'Windspeed')
    animal, antibiotic = titles[i]
    fig_ax.set_title(f'{animal} - {antibiotic}', size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(3, 6, figsize=(24, 12))
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

    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
       
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of AMR and Windspeed', size = 13, weight = 'bold')
plt.tight_layout()
# plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_WS_AMR.png', dpi = 600)
plt.show()

figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, animal_ab_df_cleaned_filled_smoothed.iloc[:,i].values, Precipitation[3:-1].values, L_ranges[i],
                          3, tau_all[i], 4, E_all[i], 'AMR', 'Precipitation')
    animal, antibiotic = titles[i]
    fig_ax.set_title(f'{animal} - {antibiotic}', size = 11, weight = 'bold')
    fig_ax.grid('--', c='grey', alpha=0.4)
    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(3, 6, figsize=(24, 12))
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

    ax[i].set_title(source_ax.get_title(loc='center'), size = 11, weight = 'bold')
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
       
fig.supxlabel('Time Series Length [L]')
fig.supylabel(r'CCM Skill (ρ)')
fig.suptitle('Convergent Cross Mapping of AMR and Precipitation', size = 13, weight = 'bold')
plt.tight_layout()
# plt.savefig(r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\CCM_Prec_AMR.png', dpi = 600)
plt.show()

# %% CCM for MDR profiles

