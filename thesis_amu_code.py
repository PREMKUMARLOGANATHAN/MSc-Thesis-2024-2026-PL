# %% Author Note

'''

This code is written by Prem Kumar Loganathan for the MSc Thesis titled, 
'Investigating Climate-driven Causal Relationships of Antimicrobial Usage and Resistance in Food-Producing Animals'

All the class objects and user-defined functions needed for this code is made available in the same repository. 
No explanation will be provided anywhere in code until and unless necessary.

'''

# %% Importing Libraries

# 1. Pre-defined Functions

import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import skew
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

# 1) AMU Animals
amu_data_path = 'D:/Antibiotics_Usedata/AMU_DF.csv'
amu_df = pd.read_csv(amu_data_path)

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

def coordinates_plot(x, color_label, *args):
    
    fig, ax = plt.subplots(figsize = (9,5))

    for i in range(np.shape(x)[0]):

        color = cm.tab10(c_l[i])
        ax.plot(x[i], color = color, alpha = 0.5, linewidth = 2)

    ax.set_xmargin(0)
    ax.set_xlabel('Time [Month]')
    ax.set_ylabel('Normalised Total Active Substance')
    ax.legend()
    ax.set_title(f'Coordinates Plot (optimal clusters = {np.max(c_l + 1)})', loc = 'left')
    ax.grid()
    plt.show()

# %% Preprocessing AMU Data
animals = ['Calves', 'Pigs', 'Poultry']
ab_name = {
    'Amoxicilline': 'Amoxicillin',
    'Amoxicilline_Clavulaanzuur': 'Amoxicillin_ClavulanicAcid',
    'Ampicilline': 'Ampicillin',
    'Apramycine': 'Apramycin',
    'Benzylpenicilline': 'Benzylpenicillin',
    'Benzylpenicilline_neomycine': 'Benzylpenicillin_Neomycin',
    'Cefalexine': 'Cephalexin',
    'Cefquinome': 'Cefquinome',
    'Ceftiofur': 'Ceftiofur',
    'Chloortetracycline': 'Chlortetracycline',
    'Colistine': 'Colistin',
    'Doxycycline': 'Doxycycline',
    'Enrofloxacine': 'Enrofloxacin',
    'Fenoxymethylpenicilline': 'Phenoxymethylpenicillin',
    'Florfenicol': 'Florfenicol',
    'Flumequine': 'Flumequine',
    'Gamithromycine': 'Gamithromycin',
    'Gentamicine': 'Gentamicin',
    'Lincomycine': 'Lincomycin',
    'Lincomycine_spectinomycine': 'Lincomycin_Spectinomycin',
    'Marbofloxacine': 'Marbofloxacin',
    'Neomycine': 'Neomycin',
    'Oxytetracycline': 'Oxytetracycline',
    'Paromomycine': 'Paromomycin',
    'Spectinomycine': 'Spectinomycin',
    'Tiamuline': 'Tiamulin',
    'Tildipirosine': 'Tildipirosin',
    'Tilmicosine': 'Tilmicosin',
    'Trim_sulfa': 'Trimethoprim_Sulfonamide',
    'Tulathromycine': 'Tulathromycin',
    'Tylosine': 'Tylosin',
    'Tylvalosine': 'Tylvalosin',
    'Cefazoline': 'Cefazolin',
    'Penethamaat': 'Penethamate',
    'Thiamfenicol': 'Thiamphenicol',
    'Cefalonium': 'Cefalonium',
    'Benzylpenicilline_streptomycine': 'Benzylpenicillin_Streptomycin',
    'Cloxacilline': 'Cloxacillin'}

ab_class = {
    # Penicillins
    'Amoxicillin': 'Penicillin',
    'Amoxicillin_ClavulanicAcid': 'Penicillin',
    'Ampicillin': 'Penicillin',
    'Benzylpenicillin': 'Penicillin',
    'Benzylpenicillin_Neomycin': 'Penicillin',
    'Phenoxymethylpenicillin': 'Penicillin',
    'Penethamate': 'Penicillin',
    'Benzylpenicillin_Streptomycin': 'Penicillin',
    'Cloxacillin': 'Penicillin',

    # Cephalosporins
    'Cephalexin': 'Cephalosporin',
    'Cefquinome': 'Cephalosporin',
    'Ceftiofur': 'Cephalosporin',
    'Cefazolin': 'Cephalosporin',
    'Cefalonium': 'Cephalosporin',

    # Tetracyclines
    'Chlortetracycline': 'Tetracycline',
    'Doxycycline': 'Tetracycline',
    'Oxytetracycline': 'Tetracycline',

    # Aminoglycosides
    'Apramycin': 'Aminoglycoside',
    'Gentamicin': 'Aminoglycoside',
    'Neomycin': 'Aminoglycoside',
    'Paromomycin': 'Aminoglycoside',
    'Spectinomycin': 'Aminoglycoside',

    # Fluoroquinolones
    'Enrofloxacin': 'Fluoroquinolone',
    'Marbofloxacin': 'Fluoroquinolone',
    'Flumequine': 'Fluoroquinolone',

    # Macrolides
    'Gamithromycin': 'Macrolide',
    'Tildipirosin': 'Macrolide',
    'Tilmicosin': 'Macrolide',
    'Tulathromycin': 'Macrolide',
    'Tylosin': 'Macrolide',
    'Tylvalosin': 'Macrolide',

    # Lincosamides
    'Lincomycin': 'Lincosamide',
    'Lincomycin_Spectinomycin': 'Lincosamide',

    # Pleuromutilins
    'Tiamulin': 'Pleuromutilin',

    # Polymyxins
    'Colistin': 'Polymyxin',

    # Phenicols
    'Florfenicol': 'Phenicols',
    'Thiamphenicol': 'Phenicols',

    # Sulfonamide
    'Trimethoprim_Sulfonamide': 'Sulfonamide'}

animal_name_map = {'PIG': 'Pigs',
                   'VECLF': 'Calves',
                   'PLTR': 'Poultry'}

amu_df['Active_Substance'] = amu_df['Active_Substance'].replace(ab_name)
amu_df['AnimalType'] = amu_df['AnimalType'].replace(animal_name_map)

amu_df_for_clustering = amu_df.groupby(['YY-MM', 'Active_Substance', 'AnimalType'])[['Total_Active_Substance']].sum().reset_index()
table_for_clustering = pd.pivot_table(data = amu_df_for_clustering, index = 'YY-MM', values = 'Total_Active_Substance', columns = ['AnimalType', 'Active_Substance'])

cols_to_drop = [col for col in table_for_clustering.columns if has_consecutive_nans(table_for_clustering[col], 5)]
amu_df_cleaned = table_for_clustering.drop(columns = cols_to_drop)
loss = ((table_for_clustering.shape[1] - amu_df_cleaned.shape[1]) / table_for_clustering.shape[1]) * 100
print(f'Loss: {np.round(loss, 2)}%')
amu_df_filled_wi = amu_df_cleaned.interpolate()

# %% Exploratory Data Analysis

# Sample AS with gaps

columns_with_gaps = [('Calves', 'Colistin'), ('Pigs','Enrofloxacin'),('Poultry','Enrofloxacin')]
fig, ax = plt.subplots(3, 1, sharex = True, figsize = (10, 7))
ax = ax.flatten()

for i, j in enumerate(columns_with_gaps):
    ax[i].plot(amu_df_cleaned.index, amu_df_cleaned[j].interpolate(), '.', c = 'red', lw = 1, markersize = 2.5, label = 'Imputed')
    ax[i].plot(amu_df_cleaned.index, amu_df_cleaned[j], '.-', c = 'steelblue', lw = 1.2, markersize = 3, label = 'Original')
    animal, antibiotic = j
    ax[i].set_title(f'{animal} - {antibiotic}', weight = 'bold', size = 11)
    ax[i].legend() if i == 0 else None
    ax[i].yaxis.set_major_locator(MaxNLocator(4))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0,0))
    formatter.set_useOffset(False)
    ax[i].yaxis.set_major_formatter(formatter)
    ax[i].set_xticks(amu_df_cleaned.index[4::6])
    ax[i].tick_params('x', rotation = 90)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    
fig.supxlabel('Time [Months]')
fig.supylabel('Active Substance Consumed (mg)')
plt.tight_layout()
plt.savefig('AMU_Animals_with gaps.png', dpi = 600)   

# Animal - wise AMU visualisation

fig, ax = plt.subplots(3, 1, sharex = True, figsize = (10, 7))
ax = ax.flatten()

for i, j in enumerate(animals):
    ax[i].plot(amu_df_filled_wi.index, amu_df_filled_wi[j].mean(axis = 1), '.-', c = 'steelblue', lw = 1.2, markersize = 3)
    ax[i].set_title(j, weight = 'bold', size = 11)
    ax[i].yaxis.set_major_locator(MaxNLocator(4))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0,0))
    formatter.set_useOffset(False)
    ax[i].yaxis.set_major_formatter(formatter)
    ax[i].set_xticks(amu_df_cleaned.index[4::6])
    ax[i].tick_params('x', rotation = 90)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    
fig.supxlabel('Time [Months]')
fig.supylabel('Active Substance Consumed (mg)')
fig.suptitle('Antimicrobial Usage Patterns in Food Producing Animals in Belgium (2017-2024)', size = 12, weight = 'bold')
plt.tight_layout()
plt.savefig('AMU_Animals_wise.png', dpi = 600)   

# Yearly consumption levels - Overall

year = amu_df_filled_wi.index.str[:4]
yearly_stats = []

for yr, df in amu_df_filled_wi.groupby(year):
    values = df.to_numpy().flatten()
    yearly_stats.append({'Year': yr, 'Mean': values.mean(), 'Std': values.std()})

yearly_stats = pd.DataFrame(yearly_stats).set_index('Year')

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(yearly_stats.index.astype(str), yearly_stats['Mean'],
    color='steelblue', edgecolor='black', width=0.7)
ax.set_title('Average Antimicrobial Usage in Belgium', fontsize=12, weight='bold')
ax.set_xlabel('Time [Year]', fontsize=12)
ax.set_ylabel('Total Active Substance Consumed (mg)', fontsize=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('Overal AMU.png', dpi = 600)
plt.show()

# Animal - wise yearly consumption pattern

year = amu_df_filled_wi.index.str[:4]
yearly_animal_stats = {}

for animal in amu_df_filled_wi.columns.levels[0]:
    df = amu_df_filled_wi[animal]
    stats = []

    for yr, group in df.groupby(year):
        values = group.to_numpy().flatten()
        stats.append({'Year': yr, 'Mean': values.mean(), 'Std': values.std(ddof=1)})
    yearly_animal_stats[animal] = pd.DataFrame(stats).set_index('Year')

animals = list(yearly_animal_stats.keys())

for animal, df in yearly_animal_stats.items():

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(df.index.astype(str), df['Mean'],
        color='steelblue', edgecolor='black', width=0.7)

    ax.set_title(f'Average Antimicrobial Usage in {animal} in Belgium', fontsize=12, weight='bold')
    ax.set_xlabel('Time [Year]', fontsize=12)
    ax.set_ylabel('Total Active Substance Consumed (mg)', fontsize=12)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f'AMU_{animal}.png', dpi=600)
    plt.show()

# Distribution - Overall consumption

overall_values = amu_df_filled_wi.values.flatten()
overall_skew = skew(overall_values)

fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(overall_values, bins=25, kde=True, ax=ax, color='steelblue')
ax.set_title('Distribution of Overal Antimicrobial Usage', fontsize=12, weight='bold')
ax.set_xlabel('Total Active Substance Consumed (mg)')
ax.set_ylabel('Frequency')
ax.text(0.95, 0.95, f'Skewness = {overall_skew:.2f}', transform=ax.transAxes,
    ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.grid(alpha=0.3, linestyle='--')
sns.despine()

plt.tight_layout()
plt.savefig('Dist_Overall_AMU.png', dpi=600)
plt.show()

# Animal - wise distribution

animals = list(amu_df_filled_wi.columns.levels[0])

for animal in animals:

    values = amu_df_filled_wi[animal].values.ravel()
    animal_skew = skew(values)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(values, bins=50, kde=True, ax=ax, color='steelblue')
    ax.set_title(f'Distribution of AMU in {animal}', fontsize=12, weight='bold')
    ax.set_xlabel('Total Active Substance Consumed (mg)')
    ax.set_ylabel('Frequency')
    ax.text(0.95, 0.95, f'Skewness = {animal_skew:.2f}', transform=ax.transAxes,
        ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(alpha=0.3, linestyle='--')
    sns.despine()

    plt.tight_layout()
    plt.savefig(f'Dist_AMU_{animal}.png', dpi=600)
    plt.show()

# %% CCM with noise

# 1) Data Creation

pigs_without_smoothing = amu_df_filled_wi['Pigs']
calves_without_smoothing = amu_df_filled_wi['Calves']
poultry_without_smoothing = amu_df_filled_wi['Poultry']

# 2) Mutual Information

mutual_info = tdmi.tdmi(preprocessing(calves_without_smoothing.mean(axis = 1)), 9, 4)
fig, ax = plt.subplots(figsize = (8, 6))

ax.plot(np.arange(1, 10, 1), mutual_info, 'o-', color = 'steelblue', lw = 1.2)
ax.grid('--', alpha = 0.4)
ax.set_xlabel(r'Time Delay ($\tau$)')
ax.set_ylabel('Mutual Information')
ax.set_title('Average Mutual Information - AMU Calves (Raw Data)', weight = 'bold')
plt.savefig('MI_calves.png', dpi = 600)

mutual_info = tdmi.tdmi(preprocessing(pigs_without_smoothing.mean(axis = 1)), 9, 4)
fig, ax = plt.subplots(figsize = (8, 6))

ax.plot(np.arange(1, 10, 1), mutual_info, 'o-', color = 'steelblue', lw = 1.2)
ax.grid('--', alpha = 0.4)
ax.set_xlabel(r'Time Delay ($\tau$)')
ax.set_ylabel('Mutual Information')
ax.set_title('Average Mutual Information - AMU Pigs (Raw Data)', weight = 'bold')
plt.savefig('MI_pigs.png', dpi = 600)

mutual_info = tdmi.tdmi(preprocessing(poultry_without_smoothing.mean(axis = 1)), 9, 4)
fig, ax = plt.subplots(figsize = (8, 6))

ax.plot(np.arange(1, 10, 1), mutual_info, 'o-', color = 'steelblue', lw = 1.2)
ax.grid('--', alpha = 0.4)
ax.set_xlabel(r'Time Delay ($\tau$)')
ax.set_ylabel('Mutual Information')
ax.set_title('Average Mutual Information - AMU Poultry (Raw Data)', weight = 'bold')
plt.savefig('MI_poultry.png', dpi = 600)

# Temperature

mutual_info = tdmi.tdmi(temp.values, 9, 4)
fig, ax = plt.subplots(figsize = (8, 6))

ax.plot(np.arange(1, 10, 1), mutual_info, 'o-',lw = 1.2, color = 'steelblue')
ax.set_xlabel('Time Delay (τ)')
ax.set_ylabel('Mutual Information')
ax.set_title('Average Mutual Information - Temperature', size =11, weight = 'bold')
ax.grid('--', alpha = 0.3)

plt.tight_layout()
plt.savefig('Temp_raw_MI.png', dpi = 600)
plt.show()

# 3) Cao's FNN

fig, ax = plt.subplots(figsize = (8, 6))
max_E = 11
opt_E = []
for e in np.arange(1, max_E):

    r = afn.afn(preprocessing(calves_without_smoothing.mean(axis = 1)), e, 8, 'euclidean', 1, None)
    opt_E.append(np.asarray(r).T)
        
E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
ax.plot(np.arange(1, max_E-1), E1, 'o-')
ax.grid('--', alpha = 0.4,)
ax.set_title("Cao's FNN method - AMU Calves (Raw Data)", weight = 'bold')
ax.set_xlabel('Embedding Dimension (E)')
ax.set_ylabel('E1 Score')
plt.tight_layout()
plt.savefig('FNN_calves.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(figsize = (8, 6))
max_E = 11
opt_E = []
for e in np.arange(1, max_E):

    r = afn.afn(preprocessing(pigs_without_smoothing.mean(axis = 1)), e, 6, 'euclidean', 1, None)
    opt_E.append(np.asarray(r).T)
        
E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
ax.plot(np.arange(1, max_E-1), E1, 'o-')
ax.grid('--', alpha = 0.4,)
ax.set_title("Cao's FNN method - AMU Pigs (Raw Data)", weight = 'bold')
ax.set_xlabel('Embedding Dimension (E)')
ax.set_ylabel('E1 Score')
plt.tight_layout()
plt.savefig('FNN_pigs.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(figsize = (8, 6))
max_E = 11
opt_E = []
for e in np.arange(1, max_E):

    r = afn.afn(preprocessing(poultry_without_smoothing.mean(axis = 1)), e, 8, 'euclidean', 1, None)
    opt_E.append(np.asarray(r).T)
        
E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
ax.plot(np.arange(1, max_E-1), E1, 'o-')
ax.grid('--', alpha = 0.4,)
ax.set_title("Cao's FNN method - AMU Poultry (Raw Data)", weight = 'bold')
ax.set_xlabel('Embedding Dimension (E)')
ax.set_ylabel('E1 Score')
plt.tight_layout()
plt.savefig('FNN_poultry.png', dpi = 600)
plt.show()

# Temperature

fig, ax = plt.subplots(figsize = (8, 6))
max_E = 11
opt_E = []
for e in np.arange(1, max_E):

    r = afn.afn(preprocessing(temp.values), e, 4, 'euclidean', 1, None)
    opt_E.append(np.asarray(r).T)
        
E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
ax.plot(np.arange(1, max_E-1), E1, 'o-')
ax.grid('--', alpha = 0.4,)
ax.set_title("Cao's FNN method - Temperature (Raw Data)", weight = 'bold')
ax.set_xlabel('Embedding Dimension (E)')
ax.set_ylabel('E1 Score')
plt.tight_layout()
plt.savefig('FNN_temp.png', dpi = 600)
plt.show()

# 4) Shadow Manifold

fig, ax = plt.subplots(figsize = (8, 6), subplot_kw = dict(projection = '3d'))
M = mv.build_shadow(calves_without_smoothing.mean(axis = 1), 6, 8)
ax.plot(M[:,0], M[:,1], M[:,2], lw = 1.2)

ax.set_xlabel(r'AMU$_{\mathrm{Calves}}$(t)')
ax.set_ylabel(r'AMU$_{\mathrm{Calves}}$(t-τ)')
ax.set_zlabel(r'AMU$_{\mathrm{Calves}}$(t-2τ)')

ax.set_title('Phase Space Reconstruction - Calves', fontweight='bold')

plt.tight_layout()
plt.savefig('calves_sm.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(figsize = (8, 6), subplot_kw = dict(projection = '3d'))
M = mv.build_shadow(calves_without_smoothing.mean(axis = 1), 4, 6)
ax.plot(M[:,0], M[:,1], M[:,2], lw = 1.2)

ax.set_xlabel(r'AMU$_{\mathrm{Pigs}}$(t)')
ax.set_ylabel(r'AMU$_{\mathrm{Pigs}}$(t-τ)')
ax.set_zlabel(r'AMU$_{\mathrm{Pigs}}$(t-2τ)')

ax.set_title('Phase Space Reconstruction - Pigs', fontweight='bold')

plt.tight_layout()
plt.savefig('pigs_sm.png', dpi = 600)
plt.show()

fig, ax = plt.subplots(figsize = (8, 6), subplot_kw = dict(projection = '3d'))
M = mv.build_shadow(poultry_without_smoothing.mean(axis = 1), 5, 7)
ax.plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
ax.set_xlabel(r'AMU$_{\mathrm{Poultry}}$(t)')
ax.set_ylabel(r'AMU$_{\mathrm{Poultry}}$(t-τ)')
ax.set_zlabel(r'AMU$_{\mathrm{Poultry}}$(t-2τ)')

ax.set_title('Phase Space Reconstruction - Poultry', fontweight='bold')

plt.tight_layout()
plt.savefig('poulty_sm.png', dpi = 600)
plt.show()

# Temperature

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='3d'))

M = mv.build_shadow(temp.values, 6, 4)
ax.plot(M[:, 0], M[:, 1], M[:, 2], lw=1.2)

ax.set_xlabel(r'Temp(t)')
ax.set_ylabel(r'Temp(t-$\tau$)')
ax.set_zlabel(r'Temp(t-2$\tau$)')

# Change the viewing angle
ax.view_init(elev=30, azim=45)

ax.set_title('Phase Space Reconstruction - Temperature', fontweight='bold')

plt.tight_layout()
plt.savefig('temp_sm.png', dpi=600)
plt.show()

# 4) CCM

L_range = np.arange(53, 94, 5)
ax = ccm_result_1(CCM.CCM, preprocessing(calves_without_smoothing.mean(axis = 1).values), preprocessing(temp.values), 
                  L_range, 4, 8, 6, 6, 'AMU Calves','Temperature')
ax.figure.tight_layout()
ax.figure.savefig('CCM_Calves_Temperature.png', dpi=600, bbox_inches='tight')
plt.show()

L_range = np.arange(28, 94, 5)
ax = ccm_result_1(CCM.CCM, preprocessing(pigs_without_smoothing.mean(axis = 1).values), preprocessing(temp.values), 
                  L_range, 4, 6, 6, 4, 'AMU Pigs', 'Temperature')
ax.figure.tight_layout()
ax.figure.savefig('CCM_Pigs_Temperature.png', dpi=600, bbox_inches='tight')
plt.show()

L_range = np.arange(43, 94, 5)
ax = ccm_result_1(CCM.CCM, preprocessing(poultry_without_smoothing.mean(axis = 1).values), preprocessing(temp.values), 
                  L_range, 4, 5, 6, 7, 'AMU Poultry', 'Temperature')
ax.figure.tight_layout()
ax.figure.savefig('CCM_Poultry_Temperature.png', dpi=600, bbox_inches='tight')
plt.show()

# %% Smoothing Climate Variables

Temperature = pd.DataFrame(temp.values).rolling(window = 5, center = True).mean().dropna()
Windspeed = pd.DataFrame(ws.values).rolling(window = 3, center = True).mean().dropna()
RH = pd.DataFrame(rh.values).rolling(window = 5, center = True).mean().dropna()
Precipitation = pd.DataFrame(prec.values).rolling(window = 3, center = True).mean().dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

variables = [(temp.values, Temperature.values, 'Temperature', 'tab:red'),
    (ws.values, Windspeed.values, 'Wind Speed', 'tab:blue'),
    (rh.values, RH.values, 'Relative Humidity', 'tab:green'),
    (prec.values, Precipitation.values, 'Precipitation', 'tab:purple')]

for ax, (original, smooth, title, color) in zip(axes.ravel(), variables):

    ax.plot(full_temp['YY-MM'][2:], original, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Original')
    if title == 'Wind Speed' or title == 'Precipitation':
        ax.plot(full_temp['YY-MM'][3:-1], smooth,
                color=color, linewidth=2, label='Smoothed(W=3)')
        if title == 'Wind Speed':
            ax.set_ylabel('Windspeed (m/s)')
        else:
            ax.set_ylabel('Precipitation (mm/day)')
    else:
        ax.plot(full_temp['YY-MM'][4:-2], smooth,
                color=color, linewidth=2, label='Smmothed(W=5)')
        if title == 'Temperature':
            ax.set_ylabel('Temperature (°C)')
        else:
            ax.set_ylabel('Humidity (%)')
    ax.set_title(title, size = 12, weight = 'bold')
    ax.grid('--', color = 'grey', alpha = 0.3)
    ax.legend()
    ax.set_xmargin(0)
    ax.set_xticks(full_temp['YY-MM'][4::6])
    ax.tick_params('x', rotation = 90)
    
fig.suptitle('Climatic Variables Before and After Smoothing',)
fig.supxlabel('Time [Months]')
plt.tight_layout()
plt.savefig('Climate Variables.png', dpi = 600)
plt.show()

# %% Finding opitmal state space reconstruction parameters for smoothened climatic variables

climate_vars1 = pd.DataFrame({'Temperature': Temperature.squeeze(), 'RH': RH.squeeze()})
climate_vars2 = pd.DataFrame({'Windspeed': Windspeed.squeeze(), 'Precipitation': Precipitation.squeeze()})

c1_processed = preprocessing(climate_vars1)
c2_processed = preprocessing(climate_vars2)

# Investigate Parameters

# 1 x 2 ACF

fig, ax = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (10, 6))
ax = ax.flatten()

for i, j in enumerate(c1_processed.columns):
    plot_acf(c1_processed[j], lags = 25, ax = ax[i], title = j, 
             zero = False, color = 'k', alpha = None,)
    ax[i].axhline(y = 1/np.exp(1), ls = '--', color = 'k', 
                  alpha = 0.8, label = 'threshold')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].legend(loc = 'lower right')
    ax[i].set_title(j, fontsize=8)
    
fig.supylabel('Correlation Coefficient')
fig.supxlabel('Time Lags')
fig.suptitle('Auto Correlation Function')
plt.tight_layout()
plt.savefig(f'')
plt.show()

fig, ax = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (10, 6))
ax = ax.flatten()

for i, j in enumerate(c2_processed.columns):
    plot_acf(c2_processed[j], lags = 25, ax = ax[i], title = j, 
             zero = False, color = 'k', alpha = None,)
    ax[i].axhline(y = 1/np.exp(1), ls = '--', color = 'k', 
                  alpha = 0.8, label = 'threshold')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].legend(loc = 'lower right')
    ax[i].set_title(j, fontsize=8)
    
fig.supylabel('Correlation Coefficient')
fig.supxlabel('Time Lags')
fig.suptitle('Auto Correlation Function')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, sharex = True, figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(c1_processed.columns):
    
    mutual_info = tdmi.tdmi(c1_processed[j], 9, 4)
    ax[i].plot(np.arange(1,10,1), mutual_info, 'o-', )
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)
    
fig.supylabel('Mutual Information')
fig.supxlabel('Time Lags')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, sharex = True, figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(c2_processed.columns):
    
    mutual_info = tdmi.tdmi(c2_processed[j], 9, 4)
    ax[i].plot(np.arange(1,10,1), mutual_info, 'o-', )
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)
    
fig.supylabel('Mutual Information')
fig.supxlabel('Time Lags')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

tau_c1 = [4, 4]
tau_c2 = [4, 4]
# 2 x 4 Cao's FNN

fig, ax = plt.subplots(1, 2, sharex = True, figsize = (12, 6))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(c1_processed.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(c1_processed[j], e, tau_c1[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-')
    # ax[i].axhline(y = 0.9, ls = '--', label = 'threshold', color = 'grey')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)
    
fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, sharex = True, figsize = (12, 6))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(c2_processed.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(c2_processed[j], e, tau_c2[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-')
    # ax[i].axhline(y = 0.9, ls = '--', label = 'threshold', color = 'grey')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)
    
fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

E_c1 = [4, 4]
E_c2 = [4, 4]

# 2 x 4 shadow manifold

fig, ax = plt.subplots(1, 2, subplot_kw = dict(projection = '3d'), figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(c1_processed.columns): 
    
    M = mv.build_shadow(c1_processed[j], E_c2[i], tau_c1[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, size = 9)
    
fig, ax = plt.subplots(1, 2, subplot_kw = dict(projection = '3d'), figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(c2_processed.columns): 
    
    M = mv.build_shadow(c2_processed[j], E_c2[i], tau_c2[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, size = 9)


# %% CCM without noise

# 1) Data Creation

amu_df_filled_as = amu_df_cleaned.interpolate().rolling(window = 5, center = True).mean().dropna()
pigs_after_smoothing = amu_df_filled_as['Pigs']
calves_after_smoothing = amu_df_filled_as['Calves']
poultry_after_smoothing = amu_df_filled_as['Poultry']

# 2) Mutual Information

mutual_info = tdmi.tdmi(preprocessing(pigs_after_smoothing.mean(axis=1)), 9, 4)
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(np.arange(1, 10), mutual_info, 'o-', color='steelblue', lw=1.2)
ax.grid('--', alpha=0.4)
ax.set_xlabel(r'Time Delay ($\tau$)')
ax.set_ylabel('Mutual Information')
ax.set_title('Average Mutual Information - AMU Pigs (After Smoothing)', weight='bold')
plt.tight_layout()
plt.savefig('MI_pigs_smoothed.png', dpi=600)
plt.show()

mutual_info = tdmi.tdmi(preprocessing(calves_after_smoothing.mean(axis=1)), 9, 4)
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(np.arange(1, 10), mutual_info, 'o-', color='steelblue', lw=1.2)
ax.grid('--', alpha=0.4)
ax.set_xlabel(r'Time Delay ($\tau$)')
ax.set_ylabel('Mutual Information')
ax.set_title('Average Mutual Information - AMU Calves (After Smoothing)', weight='bold')
plt.tight_layout()
plt.savefig('MI_calves_smoothed.png', dpi=600)
plt.show()

mutual_info = tdmi.tdmi(preprocessing(poultry_after_smoothing.mean(axis=1)), 9, 4)
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(np.arange(1, 10), mutual_info, 'o-', color='steelblue', lw=1.2)
ax.grid('--', alpha=0.4)
ax.set_xlabel(r'Time Delay ($\tau$)')
ax.set_ylabel('Mutual Information')
ax.set_title('Average Mutual Information - AMU Poultry (After Smoothing)', weight='bold')
plt.tight_layout()
plt.savefig('MI_poultry_smoothed.png', dpi=600)
plt.show()

# 2) Cao's FNN

fig, ax = plt.subplots(figsize=(8, 6))
max_E = 11
opt_E = []

for e in np.arange(1, max_E):
    r = afn.afn(preprocessing(calves_after_smoothing.mean(axis=1)), e, 6, 'euclidean', 1, None)
    opt_E.append(np.asarray(r).T)

E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
ax.plot(np.arange(1, max_E-1), E1, 'o-')
ax.grid('--', alpha=0.4)
ax.set_title("Cao's FNN Method - AMU Calves (After Smoothing)", weight='bold')
ax.set_xlabel('Embedding Dimension (E)')
ax.set_ylabel('E1 Score')
plt.tight_layout()
plt.savefig('FNN_calves_smoothed.png', dpi=600)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
max_E = 11
opt_E = []

for e in np.arange(1, max_E):
    r = afn.afn(preprocessing(pigs_after_smoothing.mean(axis=1)), e, 6, 'euclidean', 1, None)
    opt_E.append(np.asarray(r).T)

E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
ax.plot(np.arange(1, max_E-1), E1, 'o-')
ax.grid('--', alpha=0.4)
ax.set_title("Cao's FNN Method - AMU Pigs (After Smoothing)", weight='bold')
ax.set_xlabel('Embedding Dimension (E)')
ax.set_ylabel('E1 Score')
plt.tight_layout()
plt.savefig('FNN_pigs_smoothed.png', dpi=600)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
max_E = 11
opt_E = []

for e in np.arange(1, max_E):
    r = afn.afn(preprocessing(poultry_after_smoothing.mean(axis=1)), e, 6, 'euclidean', 1, None)
    opt_E.append(np.asarray(r).T)

E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
ax.plot(np.arange(1, max_E-1), E1, 'o-')
ax.grid('--', alpha=0.4)
ax.set_title("Cao's FNN Method - AMU Poultry (After Smoothing)", weight='bold')
ax.set_xlabel('Embedding Dimension (E)')
ax.set_ylabel('E1 Score')
plt.tight_layout()
plt.savefig('FNN_poultry_smoothed.png', dpi=600)
plt.show()

# 3) Shadow Manifold

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='3d'))
M = mv.build_shadow(calves_after_smoothing.mean(axis=1), 4, 7)

ax.plot(M[:,0], M[:,1], M[:,2], lw=1.2)
ax.set_xlabel(r'AMU$_{\mathrm{Calves}}$(t)')
ax.set_ylabel(r'AMU$_{\mathrm{Calves}}$(t-$\tau$)')
ax.set_zlabel(r'AMU$_{\mathrm{Calves}}$(t-2$\tau$)')
ax.set_title('Phase Space Reconstruction - Calves (After Smoothing)', fontweight='bold')
plt.tight_layout()
plt.savefig('calves_smoothed.png', dpi=600)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='3d'))
M = mv.build_shadow(pigs_after_smoothing.mean(axis=1), 4, 6)

ax.plot(M[:,0], M[:,1], M[:,2], lw=1.2)
ax.set_xlabel(r'AMU$_{\mathrm{Pigs}}$(t)')
ax.set_ylabel(r'AMU$_{\mathrm{Pigs}}$(t-$\tau$)')
ax.set_zlabel(r'AMU$_{\mathrm{Pigs}}$(t-2$\tau$)')
ax.set_title('Phase Space Reconstruction - Pigs (After Smoothing)', fontweight='bold')
plt.tight_layout()
plt.savefig('pigs_smoothed.png', dpi=600)
plt.show()


fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='3d'))
M = mv.build_shadow(poultry_after_smoothing.mean(axis=1), 4, 6)

ax.plot(M[:,0], M[:,1], M[:,2], lw=1.2)
ax.set_xlabel(r'AMU$_{\mathrm{Poultry}}$(t)')
ax.set_ylabel(r'AMU$_{\mathrm{Poultry}}$(t-$\tau$)')
ax.set_zlabel(r'AMU$_{\mathrm{Poultry}}$(t-2$\tau$)')
ax.set_title('Phase Space Reconstruction - Poultry (After Smoothing)', fontweight='bold')
plt.tight_layout()
plt.savefig('poultry_smoothed.png', dpi=600)
plt.show()

# 4) CCM

L_range = np.arange(28, 90, 5)
ax = ccm_result_1( CCM.CCM,
    preprocessing(calves_after_smoothing.mean(axis=1).values),
    preprocessing(temp.rolling(window = 5, center = True).mean().dropna().values),
    L_range, 4, 6, 4, 4, 'AMU Calves', 'Temperature')
ax.figure.tight_layout()
ax.figure.savefig('CCM_Calves_Temperature_smoothed.png', dpi=600, bbox_inches='tight')
plt.show()

L_range = np.arange(28, 90, 5)
ax = ccm_result_1(CCM.CCM,
    preprocessing(pigs_after_smoothing.mean(axis=1).values),
    preprocessing(temp.rolling(window = 5, center = True).mean().dropna().values),
    L_range, 4, 6, 4, 4, 'AMU Pigs', 'Temperature')
ax.figure.tight_layout()
ax.figure.savefig('CCM_Pigs_Temperature_smoothed.png', dpi=600, bbox_inches='tight')
plt.show()

L_range = np.arange(28, 90, 5)
ax = ccm_result_1(CCM.CCM,
    preprocessing(poultry_after_smoothing.mean(axis=1).values),
    preprocessing(temp.rolling(window = 5, center = True).mean().dropna().values), 
    L_range, 4, 6, 4, 4, 'AMU Poultry', 'Temperature')
ax.figure.tight_layout()
ax.figure.savefig('CCM_Poultry_Temperature_smoothed.png', dpi=600, bbox_inches='tight')
plt.show()
# %% Clustering - Animal Wise

amu_df_filled = preprocessing(amu_df_cleaned.interpolate().rolling(window = 5, center = True).mean().dropna())
veal_df_filled = amu_df_filled['Calves']
pig_df_filled = amu_df_filled['Pigs']
pltr_df_filled = amu_df_filled['Poultry']

# Calves

# 1) k - means

k_vals = np.arange(2, 15, 1)
sse = []
shilloutte = []
centers = []
labels = []

for i in k_vals:
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=20)
    kmeans.fit(veal_df_filled.T)
    labels.append(kmeans.labels_)
    centers.append(kmeans.cluster_centers_)
    sse.append(kmeans.inertia_)
    shilloutte.append(silhouette_score(veal_df_filled.T, kmeans.labels_))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_vals, sse, 'o-')
plt.xlabel('Number of Clusters', fontweight='bold')
plt.ylabel('SSE (Inertia)', fontweight='bold')
plt.title('Elbow Method', fontweight='bold')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(k_vals, shilloutte, 'o-')
plt.xlabel('Number of Clusters', fontweight='bold')
plt.ylabel('Silhouette Score', fontweight='bold')
plt.title('Silhouette Method', fontweight='bold')
plt.grid()
plt.savefig('kmeans_calves.png', dpi = 600)
plt.show()

c = KMeans(n_clusters=4, random_state=42, n_init=20).fit(veal_df_filled.T)
c_l = c.labels_

k_cluster_result = pd.DataFrame(
    {'Columns': veal_df_filled.columns, 'Label': c_l})

c_l = np.array(c_l)
cols = veal_df_filled.columns
colors = cm.Set1.colors

fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
ax = ax.flatten()
cluster_means_calves = {}
for i, clust in enumerate(np.unique(c_l)):

    selected_cols = cols[c_l == clust]
    data = veal_df_filled[selected_cols.values]
    cluster_means_calves[f'clust_{i + 1}'] = np.mean(data, axis=1)
    color = colors[i % len(colors)]

    for col in data.columns:
        ax[i].plot(data.index, data[col], alpha=0.15, color=color,)

    ax[i].plot(data.index, np.mean(data, axis=1), alpha=0.9,
               color=color, label=f'Cluster {i + 1} mean')
    ax[i].set_xmargin(0)
    ax[i].set_xticks(data.index[4::6])
    ax[i].tick_params(axis='x', rotation=90)
    ax[i].legend(loc='upper right')
    ax[i].set_title(f'Cluster {i + 1}', size= 9, weight = 'bold')
    ax[i].grid()

fig.supylabel('Total Active Substance (Normalised)')
fig.supxlabel('Time [Month]',)
fig.suptitle('Antimicrobial Usage in Calves (k = 4)', y=0.98, weight = 'bold')
plt.tight_layout()
plt.savefig('AMU_Calves_k4.png', dpi = 600)
plt.show()

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
X_tsne = tsne.fit_transform(veal_df_filled.T)
print(tsne.kl_divergence_)

X_tsne = pd.DataFrame(X_tsne)
X_tsne.index = cols
X_tsne['clust'] = c_l

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

for i, row in X_tsne.iterrows():
    cluster = int(row['clust'])
    color = colors[cluster]

    ax.scatter(row[0], row[1], color=color, s=10)
    ax.text(row[0] + 0.02, row[1] + 0.02, s=f'{i}', color=color, fontsize=9)

clusters = sorted(X_tsne['clust'].astype(int).unique())

legend_handles = [Line2D([0], [0], marker='o', color='w',
        label=f'Cluster {c+1}', markerfacecolor=colors[c], markersize=8) for c in clusters]
ax.legend(handles=legend_handles, title='Cluster')
ax.grid(alpha=0.5)
ax.set_xlabel('First Dimension t-SNE')
ax.set_ylabel('Second Dimension t-SNE')
ax.set_title(
    't-SNE projection of AMU Calves colored based on k-means clustering',
    loc='left')
plt.tight_layout()
plt.savefig('tsne_amu_calves.png', dpi = 600)
plt.show()

# 2) Agglomerative Clustering

linkage_data = linkage(veal_df_filled.T, method='ward', metric='euclidean')

fig, ax = plt.subplots(figsize=(12, 10))

dendrogram(linkage_data, ax=ax, orientation='right', labels=veal_df_filled.columns,)

ax.set_xlabel('Distance', fontsize=12)
ax.set_ylabel('Antibiotics', fontsize=12)
ax.set_title('Agglomerative Clustering of Antimicrobial Usage in Calves', weight = 'bold')
ax.tick_params(axis='both', labelsize=10)
ax.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.3)

plt.tight_layout()
plt.savefig('AMU_calves_Hierarchical.png', dpi=600)
plt.show()

X = veal_df_filled.copy().T
silhouette_scores = []
for k in k_vals:
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(k_vals, silhouette_scores, marker='o', linewidth=2, color='steelblue')
ax.set_xlabel('Number of clusters (k)', fontsize=9)
ax.set_ylabel('Silhouette score', fontsize=9)
ax.set_title('Silhouette Score for Agglomerative Clustering', weight = 'bold', size = 10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.savefig('optk_calves_hc.png', dpi=600)
plt.show()

hierarchical_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')

labels = hierarchical_cluster.fit_predict(veal_df_filled.T)
hc_df = pd.DataFrame({'Columns': veal_df_filled.columns, 'Labels': labels })

fig, ax = plt.subplots(4, 1, figsize = (10, 8), sharex = True)
ax = ax.flatten()

for i, clust in enumerate(sorted(hc_df['Labels'].unique())):

    selected_cols = hc_df.loc[hc_df['Labels'] == clust, 'Columns']
    data = veal_df_filled[selected_cols]
    color = colors[int(clust)]

    for col in data.columns:
        ax[i].plot(data.index, data[col], color=color, alpha=0.15)
        
    ax[i].plot(data.index, data.mean(axis=1), color=color, linewidth=2, label='Cluster Mean')
    ax[i].set_title(f'Cluster {clust+1}', fontsize=11, fontweight='bold')
    ax[i].tick_params(axis='both', labelsize=10)
    ax[i].tick_params(axis='x', rotation=90)
    ax[i].set_xticks(data.index[4::6])
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].legend(fontsize=10, title_fontsize=10)

fig.suptitle('Individual Cluster Visualisation using Agglomerative Clustering in AMU Calves', weight = 'bold')
fig.supxlabel('Time [Months]', fontsize=12)
fig.supylabel('Total Active Substance (Normalised)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('HC_AMU_calves.png', dpi=600)
plt.show()

tsne = TSNE(n_components=2, perplexity=11, random_state=42)

X_tsne = tsne.fit_transform(veal_df_filled.T)

X_tsne = pd.DataFrame(X_tsne)
X_tsne.index = veal_df_filled.columns
X_tsne['clust'] = labels

fig, ax = plt.subplots(figsize=(8, 6))

for i, row in X_tsne.iterrows():

    cluster = int(row['clust'])
    color = colors[cluster]

    ax.scatter(row[0], row[1], color=color,s=25)

    ax.text(row[0] + 0.02, row[1] + 0.02, s=i, color=color, fontsize=9)

clusters = sorted(X_tsne['clust'].astype(int).unique())

legend_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=colors[c],
        markersize=8, label=f'Cluster {c+1}') for c in clusters]
ax.legend(handles=legend_handles, title='Cluster', fontsize=10, title_fontsize=10,)
ax.set_title('t-SNE Projection of AMU Calves Colored by Agglomerative Clustering', weight = 'bold')
ax.set_xlabel('First Dimension t-SNE', fontsize=12)
ax.set_ylabel('Second Dimension t-SNE', fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.grid(linestyle='--', linewidth=0.7, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_hc_amu_calves.png', dpi=600)
plt.show()

# Pigs

# 1) k - means

k_vals = np.arange(2, 19, 1)
sse = []
shilloutte = []
centers = []
labels = []

for i in k_vals:
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=20)
    kmeans.fit(pig_df_filled.T)
    labels.append(kmeans.labels_)
    centers.append(kmeans.cluster_centers_)
    sse.append(kmeans.inertia_)
    shilloutte.append(silhouette_score(pig_df_filled.T, kmeans.labels_))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_vals, sse, 'o-')
plt.xlabel('Number of Clusters', fontweight='bold')
plt.ylabel('SSE (Inertia)', fontweight='bold')
plt.title('Elbow Method', fontweight='bold')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(k_vals, shilloutte, 'o-')
plt.xlabel('Number of Clusters', fontweight='bold')
plt.ylabel('Silhouette Score', fontweight='bold')
plt.title('Silhouette Method', fontweight='bold')
plt.grid()
plt.savefig('kmeans_pigs.png', dpi = 600)
plt.show()

c = KMeans(n_clusters=5, random_state=42, n_init=20).fit(pig_df_filled.T)
c_l = c.labels_

k_cluster_result = pd.DataFrame({'Columns': pig_df_filled.columns, 'Label': c_l})

c_l = np.array(c_l)
cols = pig_df_filled.columns
colors = cm.Set1.colors

fig, ax = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
ax = ax.flatten()
cluster_means_pigs = {}
for i, clust in enumerate(np.unique(c_l)):

    selected_cols = cols[c_l == clust]
    data = pig_df_filled[selected_cols.values]
    cluster_means_pigs[f'clust_{i + 1}'] = np.mean(data, axis=1)
    color = colors[i % len(colors)]

    for col in data.columns:
        ax[i].plot(data.index, data[col], alpha=0.15, color=color,)

    ax[i].plot(data.index, np.mean(data, axis=1), alpha=0.9,
               color=color, label=f'Cluster {i + 1} mean')
    ax[i].set_xmargin(0)
    ax[i].set_xticks(data.index[4::6])
    ax[i].tick_params(axis='x', rotation=90)
    ax[i].legend(loc='upper right')
    ax[i].set_title(f'Cluster {i + 1}', size= 9, weight = 'bold')
    ax[i].grid()

fig.supylabel('Total Active Substance (Normalised)')
fig.supxlabel('Time [Month]',)
fig.suptitle('Antimicrobial Usage in Pigs (k = 5)', y=0.98, weight = 'bold')
plt.tight_layout()
plt.savefig('AMU_Pigs_k4.png', dpi = 600)
plt.show()

tsne = TSNE(n_components=2, random_state=42, perplexity=11)
X_tsne = tsne.fit_transform(pig_df_filled.T)
print(tsne.kl_divergence_)

X_tsne = pd.DataFrame(X_tsne)
X_tsne.index = cols
X_tsne['clust'] = c_l

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

for i, row in X_tsne.iterrows():
    cluster = int(row['clust'])
    color = colors[cluster]

    ax.scatter(row[0], row[1], color=color, s=10)
    ax.text(row[0] + 0.02, row[1] + 0.02, s=f'{i}', color=color, fontsize=9)

clusters = sorted(X_tsne['clust'].astype(int).unique())

legend_handles = [Line2D([0], [0], marker='o', color='w',
        label=f'Cluster {c+1}', markerfacecolor=colors[c], markersize=8) for c in clusters]
ax.legend(handles=legend_handles, title='Cluster')
ax.grid(alpha=0.5)
ax.set_xlabel('First Dimension t-SNE')
ax.set_ylabel('Second Dimension t-SNE')
ax.set_title(
    't-SNE projection of AMU Pigs colored based on k-means clustering',
    loc='left')
plt.tight_layout()
plt.savefig('tsne_amu_pigs.png', dpi = 600)
plt.show()

# 2) Agglomerative Clustering

linkage_data = linkage(pig_df_filled.T, method='ward', metric='euclidean')

fig, ax = plt.subplots(figsize=(12, 10))

dendrogram(linkage_data, ax=ax, orientation='right', labels=pig_df_filled.columns,)

ax.set_xlabel('Distance', fontsize=12)
ax.set_ylabel('Antibiotics', fontsize=12)
ax.set_title('Agglomerative Clustering of Antimicrobial Usage in Pigs', weight = 'bold')
ax.tick_params(axis='both', labelsize=10)
ax.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.3)

plt.tight_layout()
plt.savefig('AMU_pigs_Hierarchical.png', dpi=600)
plt.show()

X = pig_df_filled.copy().T

silhouette_scores = []
for k in k_vals:
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(k_vals, silhouette_scores, marker='o', linewidth=2, color='steelblue')
ax.set_xlabel('Number of clusters (k)', fontsize=9)
ax.set_ylabel('Silhouette score', fontsize=9)
ax.set_title('Silhouette Score for Agglomerative Clustering', weight = 'bold', size = 10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.savefig('optk_pigs_hc.png', dpi=600)
plt.show()

hierarchical_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')

labels = hierarchical_cluster.fit_predict(pig_df_filled.T)
hc_df = pd.DataFrame({'Columns': pig_df_filled.columns, 'Labels': labels })

fig, ax = plt.subplots(4, 1, figsize = (10, 8), sharex = True)
ax = ax.flatten()

for i, clust in enumerate(sorted(hc_df['Labels'].unique())):

    selected_cols = hc_df.loc[hc_df['Labels'] == clust, 'Columns']
    data = pig_df_filled[selected_cols]
    color = colors[int(clust)]

    for col in data.columns:
        ax[i].plot(data.index, data[col], color=color, alpha=0.15)
        
    ax[i].plot(data.index, data.mean(axis=1), color=color, linewidth=2, label='Cluster Mean')
    ax[i].set_title(f'Cluster {clust+1}', fontsize=11, fontweight='bold')
    ax[i].tick_params(axis='both', labelsize=10)
    ax[i].tick_params(axis='x', rotation=90)
    ax[i].set_xticks(data.index[4::6])
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].legend(fontsize=10, title_fontsize=10)

fig.suptitle('Individual Cluster Visualisation using Agglomerative Clustering in AMU Pigs', weight = 'bold')
fig.supxlabel('Time [Months]', fontsize=12)
fig.supylabel('Total Active Substance (Normalised)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('HC_AMU_pigs.png', dpi=600)
plt.show()

tsne = TSNE(n_components=2, perplexity=11, random_state=42)

X_tsne = tsne.fit_transform(pig_df_filled.T)

X_tsne = pd.DataFrame(X_tsne)
X_tsne.index = pig_df_filled.columns
X_tsne['clust'] = labels

fig, ax = plt.subplots(figsize=(8, 6))

for i, row in X_tsne.iterrows():

    cluster = int(row['clust'])
    color = colors[cluster]

    ax.scatter(row[0], row[1], color=color,s=25)
    ax.text(row[0] + 0.02, row[1] + 0.02, s=i, color=color, fontsize=9)

clusters = sorted(X_tsne['clust'].astype(int).unique())

legend_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=colors[c],
        markersize=8, label=f'Cluster {c+1}') for c in clusters]
ax.legend(handles=legend_handles, title='Cluster', fontsize=10, title_fontsize=10,)
ax.set_title('t-SNE Projection of AMU Pigs Colored by Agglomerative Clustering', weight = 'bold')
ax.set_xlabel('First Dimension t-SNE', fontsize=12)
ax.set_ylabel('Second Dimension t-SNE', fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.grid(linestyle='--', linewidth=0.7, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_hc_amu_pigs.png', dpi=600)
plt.show()

# 3) Poultry

# 1) k - means

k_vals = np.arange(2, 8, 1)
sse = []
shilloutte = []
centers = []
labels = []

for i in k_vals:
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=20)
    kmeans.fit(pltr_df_filled.T)
    labels.append(kmeans.labels_)
    centers.append(kmeans.cluster_centers_)
    sse.append(kmeans.inertia_)
    shilloutte.append(silhouette_score(pltr_df_filled.T, kmeans.labels_))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_vals, sse, 'o-')
plt.xlabel('Number of Clusters', fontweight='bold')
plt.ylabel('SSE (Inertia)', fontweight='bold')
plt.title('Elbow Method', fontweight='bold')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(k_vals, shilloutte, 'o-')
plt.xlabel('Number of Clusters', fontweight='bold')
plt.ylabel('Silhouette Score', fontweight='bold')
plt.title('Silhouette Method', fontweight='bold')
plt.grid()
plt.savefig('kmeans_pltr.png', dpi = 600)
plt.show()

c = KMeans(n_clusters=4, random_state=42, n_init=20).fit(pltr_df_filled.T)
c_l = c.labels_

k_cluster_result = pd.DataFrame({'Columns': pltr_df_filled.columns, 'Label': c_l})

c_l = np.array(c_l)
cols = pltr_df_filled.columns
colors = cm.Set1.colors

fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
ax = ax.flatten()
cluster_means_pltr = {}
for i, clust in enumerate(np.unique(c_l)):

    selected_cols = cols[c_l == clust]
    data = pltr_df_filled[selected_cols.values]
    cluster_means_pltr[f'clust_{i + 1}'] = np.mean(data, axis=1)
    color = colors[i % len(colors)]

    for col in data.columns:
        ax[i].plot(data.index, data[col], alpha=0.15, color=color,)

    ax[i].plot(data.index, np.mean(data, axis=1), alpha=0.9,
               color=color, label=f'Cluster {i + 1} mean')
    ax[i].set_xmargin(0)
    ax[i].set_xticks(data.index[4::6])
    ax[i].tick_params(axis='x', rotation=90)
    ax[i].legend(loc='upper right')
    ax[i].set_title(f'Cluster {i + 1}', size= 9, weight = 'bold')
    ax[i].grid()

fig.supylabel('Total Active Substance (Normalised)')
fig.supxlabel('Time [Month]',)
fig.suptitle('Antimicrobial Usage in Poultry (k = 4)', y=0.98, weight = 'bold')
plt.tight_layout()
plt.savefig('AMU_Poultry_k4.png', dpi = 600)
plt.show()

tsne = TSNE(n_components=2, random_state=42, perplexity=4)
X_tsne = tsne.fit_transform(pltr_df_filled.T)
print(tsne.kl_divergence_)

X_tsne = pd.DataFrame(X_tsne)
X_tsne.index = cols
X_tsne['clust'] = c_l

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

for i, row in X_tsne.iterrows():
    cluster = int(row['clust'])
    color = colors[cluster]

    ax.scatter(row[0], row[1], color=color, s=10)
    ax.text(row[0] + 0.02, row[1] + 0.02, s=f'{i}', color=color, fontsize=9)

clusters = sorted(X_tsne['clust'].astype(int).unique())

legend_handles = [Line2D([0], [0], marker='o', color='w',
        label=f'Cluster {c+1}', markerfacecolor=colors[c], markersize=8) for c in clusters]
ax.legend(handles=legend_handles, title='Cluster')
ax.grid(alpha=0.5)
ax.set_xlabel('First Dimension t-SNE')
ax.set_ylabel('Second Dimension t-SNE')
ax.set_title('t-SNE projection of AMU Poultry colored based on k-means clustering', loc='left')
plt.tight_layout()
plt.savefig('tsne_amu_pltr.png', dpi = 600)
plt.show()

# 2) Agglomerative Clustering

linkage_data = linkage(pltr_df_filled.T, method='ward', metric='euclidean')

fig, ax = plt.subplots(figsize=(12, 10))

dendrogram(linkage_data, ax=ax, orientation='right', labels=pltr_df_filled.columns,)

ax.set_xlabel('Distance', fontsize=12)
ax.set_ylabel('Antibiotics', fontsize=12)
ax.set_title('Agglomerative Clustering of Antimicrobial Usage in Poultry', weight = 'bold')
ax.tick_params(axis='both', labelsize=10)
ax.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.3)

plt.tight_layout()
plt.savefig('AMU_poultry_Hierarchical.png', dpi=600)
plt.show()

X = pltr_df_filled.copy().T

silhouette_scores = []
for k in k_vals:
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(k_vals, silhouette_scores, marker='o', linewidth=2, color='steelblue')
ax.set_xlabel('Number of clusters (k)', fontsize=9)
ax.set_ylabel('Silhouette score', fontsize=9)
ax.set_title('Silhouette Score for Agglomerative Clustering', weight = 'bold', size = 10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.savefig('optk_pltr_hc.png', dpi=600)
plt.show()

hierarchical_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')

labels = hierarchical_cluster.fit_predict(pltr_df_filled.T)
hc_df = pd.DataFrame({'Columns': pltr_df_filled.columns, 'Labels': labels })

fig, ax = plt.subplots(4, 1, figsize = (10, 8), sharex = True)
ax = ax.flatten()

for i, clust in enumerate(sorted(hc_df['Labels'].unique())):

    selected_cols = hc_df.loc[hc_df['Labels'] == clust, 'Columns']
    data = pltr_df_filled[selected_cols]
    color = colors[int(clust)]

    for col in data.columns:
        ax[i].plot(data.index, data[col], color=color, alpha=0.15)
        
    ax[i].plot(data.index, data.mean(axis=1), color=color, linewidth=2, label='Cluster Mean')
    ax[i].set_title(f'Cluster {clust+1}', fontsize=11, fontweight='bold')
    ax[i].tick_params(axis='both', labelsize=10)
    ax[i].tick_params(axis='x', rotation=90)
    ax[i].set_xticks(data.index[4::6])
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].legend(fontsize=10, title_fontsize=10)

fig.suptitle('Individual Cluster Visualisation using Agglomerative Clustering in AMU Poultry', weight = 'bold')
fig.supxlabel('Time [Months]', fontsize=12)
fig.supylabel('Total Active Substance (Normalised)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('HC_AMU_pltr.png', dpi=600)
plt.show()

tsne = TSNE(n_components=2, perplexity=4, random_state=42)

X_tsne = tsne.fit_transform(pltr_df_filled.T)

X_tsne = pd.DataFrame(X_tsne)
X_tsne.index = pltr_df_filled.columns
X_tsne['clust'] = labels

fig, ax = plt.subplots(figsize=(8, 6))

for i, row in X_tsne.iterrows():

    cluster = int(row['clust'])
    color = colors[cluster]
    ax.scatter(row[0], row[1], color=color,s=25)
    ax.text(row[0] + 0.02, row[1] + 0.02, s=i, color=color, fontsize=9)

clusters = sorted(X_tsne['clust'].astype(int).unique())

legend_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=colors[c],
        markersize=8, label=f'Cluster {c+1}') for c in clusters]
ax.legend(handles=legend_handles, title='Cluster', fontsize=10, title_fontsize=10,)
ax.set_title('t-SNE Projection of AMU Poultry Colored by Agglomerative Clustering', weight = 'bold')
ax.set_xlabel('First Dimension t-SNE', fontsize=12)
ax.set_ylabel('Second Dimension t-SNE', fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.grid(linestyle='--', linewidth=0.7, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_hc_amu_pltr.png', dpi=600)
plt.show()

# %% Cluserting - Overall AMU - goes if section 4.4.3 is removed

k_vals = np.arange(2, 19, 1)
sse = []
shilloutte = []
centers = []
labels = []
ch = []

for i in k_vals:
    kmeans = KMeans(n_clusters = i, random_state = 42, n_init = 20)
    kmeans.fit(amu_df_filled.T)
    labels.append(kmeans.labels_)
    centers.append(kmeans.cluster_centers_)
    sse.append(kmeans.inertia_)
    shilloutte.append(silhouette_score(amu_df_filled.T, kmeans.labels_))
    ch.append(calinski_harabasz_score(amu_df_filled.T, kmeans.labels_)) # Dont use

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(k_vals, sse, 'o-')
plt.xlabel('Number of Clusters', fontweight = 'bold')
plt.ylabel('SSE (Inertia)', fontweight = 'bold')
plt.title('Elbow Method', fontweight = 'bold')
plt.grid()

plt.subplot(1,2,2)
plt.plot(k_vals, shilloutte, 'o-')
plt.xlabel('Number of Clusters', fontweight = 'bold')
plt.ylabel('Silhouette Score', fontweight = 'bold')
plt.title('Silhouette Method', fontweight = 'bold')
plt.grid()
# plt.savefig('Choosing Optimal Clusters for AMU.png', dpi = 600)
plt.show()

c = KMeans(n_clusters = 4, random_state = 42, n_init = 20).fit(amu_df_filled.T)
c_l = c.labels_

def coordinates_plot(x, color_label, *args):
    
    fig, ax = plt.subplots(figsize = (9,5))

    for i in range(np.shape(x)[0]):

        color = cm.tab10(c_l[i])
        ax.plot(x[i], color = color, alpha = 0.5, linewidth = 2)

    ax.set_xmargin(0)
    ax.set_xlabel('Time [Month]')
    ax.set_ylabel('Normalised Total Active Substance')
    ax.legend()
    ax.set_title(f'Coordinates Plot (optimal clusters = {np.max(c_l + 1)})', loc = 'left')
    ax.grid()
    plt.show()
        
coordinates_plot(amu_df_filled.T.values, c_l, )

c_l = np.array(c_l)
cols = amu_df_filled.columns
colors = cm.tab10.colors

fig, ax = plt.subplots(4, 1, figsize = (10, 12), sharex = True)
ax = ax.flatten()
cluster_means = {}
for i, clust in enumerate(np.unique(c_l)):
    
    selected_cols = cols[c_l == clust]
    data = amu_df_filled[selected_cols.values]
    cluster_means[f'Cluster {i + 1}'] = np.mean(data, axis=1)

    for col in data.columns:
        ax[i].plot(data.index, data[col], alpha = 0.15, color = colors[i],)

    ax[i].plot(data.index, np.mean(data, axis = 1), alpha = 0.9, color = colors[i], label = f'Cluster {i + 1} mean')
    ax[i].set_xmargin(0)
    ax[i].set_xticks(data.index[::6])
    ax[i].tick_params(axis = 'x', rotation = 90)
    ax[i].legend(loc = 'upper right')
    ax[i].set_title(f'Cluster {i + 1}', loc = 'left', size = 8.5)
    ax[i].grid()

fig.supylabel('Total Active Substance (Normalised)')
fig.supxlabel('Time [Month]')
fig.suptitle('Antimicrobial Usage in Food Producing Animals')
plt.tight_layout()
plt.show()

tsne = TSNE(n_components = 2, perplexity= 17, random_state = 42)
X_new = tsne.fit_transform(amu_df_filled.T)
print(f'KL divergence: {tsne.kl_divergence_:.4f}')

X_tsne = pd.DataFrame(X_new, columns=['TSNE1', 'TSNE2'], index=cols)
X_tsne['Cluster'] = c_l

fig, ax = plt.subplots(figsize=(10, 8))

for cluster in sorted(X_tsne['Cluster'].unique()):

    data = X_tsne[X_tsne['Cluster'] == cluster]

    ax.scatter(data['TSNE1'], data['TSNE2'], s=25,
        color=colors[int(cluster) % len(colors)], label=f'Cluster {cluster + 1}')

    for idx, row in data.iterrows():
        ax.text(row['TSNE1'] + 0.1, row['TSNE2'] + 0.1, cluster + 1, fontsize=8.5)

ax.set_title('t-SNE projection of AMU in Food Producing \nAnimals in Belgium coloured by K-means clusters', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.grid(alpha=0.3)
ax.legend(title='Cluster')

# Colored by Animal

animals = X_tsne.index.get_level_values(0)
unique_animals = animals.unique()

animal_colors = {
    animal: colors[i % len(colors)]
    for i, animal in enumerate(unique_animals)}

fig, ax = plt.subplots(figsize=(10, 8))

for animal in unique_animals:

    data = X_tsne[animals == animal]

    ax.scatter(data['TSNE1'], data['TSNE2'], s=25,
        color=animal_colors[animal], label=animal)

    for idx, row in data.iterrows():
        ax.text( row['TSNE1'] + 0.1, row['TSNE2'] + 0.1,
            str(int(row['Cluster'] + 1)), fontsize=8.5)
ax.set_title('t-SNE projection of AMU in Food Producing \nAnimals in Belgium coloured by Animal Types',
             fontsize = 12, weight = 'bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.grid(alpha=0.3)
ax.legend(title='Animal')

# Colored by ABs

antibiotics = X_tsne.index.get_level_values(1)
unique_antibiotics = antibiotics.unique()
X_tsne['AB_Class'] = antibiotics.map(
    lambda x: ab_class.get(ab_name.get(x, x), 'Other'))

classes = sorted(X_tsne['AB_Class'].unique())
class_colors = {
    cls: colors[i % len(colors)]
    for i, cls in enumerate(classes)}

fig, ax = plt.subplots(figsize=(10, 8))

for cls in classes:
    data = X_tsne[X_tsne['AB_Class'] == cls]
    ax.scatter(data['TSNE1'], data['TSNE2'], s=30, color=class_colors[cls], label=cls)
    for idx, row in data.iterrows():
        animal = idx[0]
        ab = idx[1]
        ax.text( row['TSNE1'] + 0.1, row['TSNE2'] + 0.1, row['Cluster'] + 1)

ax.set_title('t-SNE projection of AMU in Food Producing \nAnimals in Belgium coloured by Antibiotic Class',
             fontsize = 12, weight = 'bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.grid(alpha=0.3)

ax.legend(title='Antibiotic class',)

plt.tight_layout()
plt.show()

# %% Correlation Coefficient for Animal Clusters

calves_cluster_df = pd.DataFrame(cluster_means_calves)
pigs_cluster_df = pd.DataFrame(cluster_means_pigs)
pltr_cluster_df = pd.DataFrame(cluster_means_pltr)

full_df = pd.concat([calves_cluster_df, pigs_cluster_df, pltr_cluster_df], axis =1)
full_df['Temperature'] = preprocessing(Temperature).values
full_df['RH'] = preprocessing(RH).values
full_df['Windspeed'] = preprocessing(Windspeed).values[1:-1]
full_df['Precipitation'] = preprocessing(Precipitation).values[1:-1]
full_df.columns = (
    [f'Calves C{i}' for i in range(1, 5)] +
    [f'Pigs C{i}' for i in range(1, 6)] +
    [f'Poultry C{i}' for i in range(1, 5)] +
    ['Temperature', 'RH', 'Windspeed', 'Precipitation'])

corr = full_df.corr(method='pearson')
mask = np.tril(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, mask = mask, cmap = 'RdBu_r', vmin = -1, vmax = 1,
    center = 0,
    square = True, linewidths = 0.5,linecolor = 'white',
    cbar_kws = dict(shrink = 0.3, label = 'Correalation Coefficient',
                   orientation = 'horizontal', location = 'bottom', pad = 0.01), ax = ax)

ax.tick_params(axis='x', rotation=90, bottom=False,
    top=True, labelbottom=False, labeltop=True)
ax.tick_params(axis='y', rotation=0, left = False, 
    right = True, labelleft = False, labelright = True)
plt.setp(ax.get_xticklabels(), ha='center')
plt.setp(ax.get_yticklabels(), va ='center')

for i, tick in enumerate(ax.get_xticklabels()):
    if i < 4:
        tick.set_color('tab:blue')
    elif i < 9:
        tick.set_color('tab:green')
    elif i < 13:
        tick.set_color('tab:orange')
    else:
        tick.set_color('black')

for i, tick in enumerate(ax.get_yticklabels()):
    if i < 4:
        tick.set_color('tab:blue')
    elif i < 9:
        tick.set_color('tab:green')
    elif i < 13:
        tick.set_color('tab:orange')
    else:
        tick.set_color('black')

plt.tight_layout()
plt.savefig('Corr_Coefficient_Cluster_wise.png', dpi = 600)
plt.show()

# %% Animal-wise causal relationship using CCM

# 1) Data Creation

full_df = pd.concat([calves_cluster_df, pigs_cluster_df, pltr_cluster_df], axis =1)
full_df.columns = (
    [f'Calves C{i}' for i in range(1, 5)] +
    [f'Pigs C{i}' for i in range(1, 6)] +
    [f'Poultry C{i}' for i in range(1, 5)])

# 2) Visualisation

fig, ax = plt.subplots(5, 3, sharex = True, figsize = (14, 12), constrained_layout = True)
ax = ax.flatten()

for i, j in enumerate(full_df.columns):
    ax[i].plot(full_df.index, full_df[j], 'o-', lw = 1.2, markersize = 3, c = 'steelblue')
    ax[i].set_title(j, weight = 'bold')
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_xticks(full_df.index[4::6])
    ax[i].tick_params('x', labelbottom = True, rotation = 90) if i > 9 else None
    
fig.delaxes(ax[13])
fig.delaxes(ax[14])
fig.suptitle('Animal-wise cluster visualisation of AMU in Food Producing Animals in Belgium', weight = 'bold', size = 14)
fig.supxlabel('Time [Months]')
fig.supylabel('Active Substance Consumed (Normalised)')
plt.savefig('animal-wise_cluster.png', dpi = 600)
plt.show()

# 3) Mutual Information

fig, ax = plt.subplots(5, 3, sharex = True, figsize = (14, 12), constrained_layout = True)
ax = ax.flatten()

for i, j in enumerate(full_df.columns):
    mutual_info = tdmi.tdmi(full_df[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw =1.2, markersize = 3, c = 'steelblue')
    ax[i].set_title(j, weight = 'bold')
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].tick_params('x', labelbottom = True,) if i > 9 else None
    
fig.delaxes(ax[13])
fig.delaxes(ax[14])
fig.suptitle('Average Mutual Information for AMU Clusters', weight = 'bold', size = 14)
fig.supxlabel('Time Delay')
fig.supylabel('Mutual Informaion')
plt.savefig('ami_animal_wise.png', dpi = 600)
plt.show()

tau_all = [6, 8, 7, 6, 5, 6, 8, 5, 6, 8, 6, 5, 5]

# 4) Embedding Dimension

fig, ax = plt.subplots(5, 3, sharex = True, figsize = (14, 12))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(full_df.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(full_df[j], e, tau_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, weight = 'bold')
    ax[i].tick_params('x', labelbottom = True) if i > 9 else None
        
fig.delaxes(ax[13])
fig.delaxes(ax[14])    
fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension", weight = 'bold')
plt.tight_layout()
plt.savefig('caofnn_animalwise.png', dpi = 600)
plt.show()

E_all = [6, 5, 3, 5, 6, 4, 6, 6, 4, 4, 5, 5, 4]

# 5) Shadow Manifold

for i, j in enumerate(full_df.columns):
    
    fig, ax = plt.subplots(subplot_kw = dict(projection = '3d'))
    M = mv.build_shadow(full_df[j], E_all[i], tau_all[i])
    ax.plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax.set_title(j, weight = 'bold')
    if i == 0:
        ax.set_xlabel('AMU(t)')
        ax.set_ylabel(r'AMU(t-$\tau$)')
        ax.set_zlabel(r'AMU(t-2$\tau$)')
    else:
        None   
    plt.savefig(fr'D:\MSc Thesis\Codes\New folder\{j}_sm.png', dpi = 600)

# 6) CCM

titles = full_df.columns

L_ranges = [
    np.arange(43, 90, 5), np.arange(53, 90, 5),
    np.arange(28, 90, 5), np.arange(38, 90, 5),
    np.arange(38, 90, 5), np.arange(28, 90, 5),
    np.arange(53, 90, 5), np.arange(38, 90, 5),
    np.arange(28, 90, 5), np.arange(38, 90, 5),
    np.arange(38, 90, 5), np.arange(33, 90, 5), 
    np.arange(28, 90, 5)]

# 6.1) Temperature

figs = []
for i in range(13):
    fig_ax = ccm_result_1(CCM.CCM, full_df.iloc[:,i].values, c1_processed['Temperature'], L_ranges[i], 
                          tau_c1[0], tau_all[i], E_c1[0], E_all[i], 'AMU', 'Temperature')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

for i in range(13):
    fig, ax = plt.subplots(figsize = (6, 6))
    source_ax = figs[i]

    for line in source_ax.lines:
        ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker())

    ax.set_title(source_ax.get_title(loc='center'), weight = 'bold')
    ax.set_xlabel(source_ax.get_xlabel())
    ax.set_ylabel(source_ax.get_ylabel())
    ax.grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax.legend()
    plt.savefig(fr'D:\MSc Thesis\Codes\New folder (2)\{i}_temperature.png', dpi = 600)
    plt.show()

# 6.2) Relative Humidity

figs = []
for i in range(13):
    fig_ax = ccm_result_1(CCM.CCM, full_df.iloc[:,i].values, c1_processed['RH'], L_ranges[i], 
                          tau_c1[1], tau_all[i], E_c1[1], E_all[i], 'AMU', 'RH')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

for i in range(13):
    fig, ax = plt.subplots(figsize = (6, 6))
    source_ax = figs[i]

    for line in source_ax.lines:
        ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker())

    ax.set_title(source_ax.get_title(loc='center'), weight = 'bold')
    ax.set_xlabel(source_ax.get_xlabel())
    ax.set_ylabel(source_ax.get_ylabel())
    ax.grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax.legend()
    plt.savefig(fr'D:\MSc Thesis\Codes\New folder (2)\{i}_rh.png', dpi = 600)
    plt.show()

# 6.3) Windspeed

figs = []
for i in range(13):
    fig_ax = ccm_result_1(CCM.CCM, full_df.iloc[:,i].values, c2_processed['Windspeed'][1:-1], L_ranges[i], 
                          tau_c2[0], tau_all[i], E_c2[0], E_all[i], 'AMU', 'Windspeed')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

for i in range(13):
    fig, ax = plt.subplots(figsize = (6, 6))
    source_ax = figs[i]

    for line in source_ax.lines:
        ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker())

    ax.set_title(source_ax.get_title(loc='center'), weight = 'bold')
    ax.set_xlabel(source_ax.get_xlabel())
    ax.set_ylabel(source_ax.get_ylabel())
    ax.grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax.legend()
    plt.savefig(fr'D:\MSc Thesis\Codes\New folder (2)\{i}_ws.png', dpi = 600)
    plt.show()

# 6.4) Precipitation

figs = []
for i in range(13):
    fig_ax = ccm_result_1(CCM.CCM, full_df.iloc[:,i].values, c2_processed['Precipitation'][1:-1], L_ranges[i], 
                          tau_c2[1], tau_all[i], E_c2[1], E_all[i], 'AMU', 'Precipitation')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

for i in range(13):
    fig, ax = plt.subplots(figsize = (6, 6))
    source_ax = figs[i]

    for line in source_ax.lines:
        ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker())

    ax.set_title(source_ax.get_title(loc='center'), weight = 'bold')
    ax.set_xlabel(source_ax.get_xlabel())
    ax.set_ylabel(source_ax.get_ylabel())
    ax.grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax.legend()
    plt.savefig(fr'D:\MSc Thesis\Codes\New folder (2)\{i}_prec.png', dpi = 600)
    plt.show()

# %% Investigating Causal Relationship using Correlation Coefficient - goes
# Dont do after smoothing
single_df = pd.DataFrame({
    'Full AMU': amu_df_filled.mean(axis=1),
    'AMU Clust 1': cluster_means['clust_1'],
    'AMU Clust 2': cluster_means['clust_2'],
    'AMU Clust 3': cluster_means['clust_3'],
    'Temperature': preprocessing(temp.values),
    'Relative Humidity': preprocessing(rh.values),
    'Precipitation': preprocessing(prec.values),
    'Wind Speed': preprocessing(ws.values)}, index = amu_df_filled.index)

corr = single_df.corr(method='pearson')
mask = np.tril(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, mask = mask, cmap = 'RdBu_r', vmin = -1, vmax = 1,
    center = 0, annot = True, fmt = '.2f', annot_kws = {'size': 9},
    square = True, linewidths = 0.5,linecolor = 'white',
    cbar_kws = dict(shrink = 0.3, label = 'Correalation Coefficient',
                   orientation = 'horizontal', location = 'bottom', pad = 0.01), ax = ax)

rect = patches.Rectangle(xy = (4, 0), width = 4, height = 4,
                         fill = False, edgecolor = 'k', lw = 2)
ax.add_patch(rect)
ax.text(1, 2, 'AMU', color = 'k', weight = 'bold', bbox = dict(facecolor='none', edgecolor = 'k'))
ax.text(2.4, 4.3, 'AMU & \nClimate', color = 'k', weight = 'bold', bbox = dict(facecolor='none', edgecolor = 'k'))
ax.text(4.5, 6.5, 'Climate', color = 'k', weight = 'bold', bbox = dict(facecolor='none', edgecolor = 'k'))

ax.tick_params(axis='x', rotation=90, bottom=False,
    top=True, labelbottom=False, labeltop=True)
ax.tick_params(axis='y', rotation=0, left = False, 
    right = True, labelleft = False, labelright = True)
plt.setp(ax.get_xticklabels(), ha='center')
plt.setp(ax.get_yticklabels(), va ='center')
plt.tight_layout()
plt.show()



# %% Investigating Causal Relationship using CCM for 6 clusters - goes
# Putting all in single df
cluster_df = pd.DataFrame(cluster_means)
all_in_df = amu_df_filled.mean(axis = 1)
pig_all_df = amu_df_filled['Pigs'].mean(axis = 1)
pltr_all_df = amu_df_filled['Poultry'].mean(axis = 1)
calf_all_df = amu_df_filled['Calves'].mean(axis = 1)
full_df = cluster_df.copy()
# full_df = pd.concat([all_in_df.rename('All AMU'),pig_all_df.rename('Pigs'),
        # pltr_all_df.rename('Poultry'), calf_all_df.rename('Calves'), cluster_df], axis = 1)

fig, ax = plt.subplots(1, 4, figsize = (16, 5), sharex = True)
ax = ax.flatten()

for i, j in enumerate(full_df.columns):
    
    ax[i].plot(full_df.index, full_df[j], marker = 'o', lw = 1.2, color = 'tab:blue')
    ax[i].set_title(j, weight = 'bold')
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].tick_params('x', rotation = 90)
    ax[i].set_xticks(full_df.index[1::6])
    ax[i].set_xmargin(0)

fig.suptitle('Antimicrobial Usage in Food Producing Animals' , weight = 'bold')
fig.supxlabel('Time[Months]')
fig.supylabel('Active Substance Consumed (Normalised)')
plt.tight_layout()
plt.savefig('fourcluster_for_ccm.png', dpi = 600)
plt.show()

# Noisy

# 2 x 5 ACF

fig, ax = plt.subplots(2, 4, sharex = True, sharey = True, figsize = (17, 8))
ax = ax.flatten()

for i, j in enumerate(full_df.columns):
    plot_acf(full_df[j], lags = 25, ax = ax[i], title = j, 
             zero = False, color = 'k', alpha = None,)
    ax[i].axhline(y = 1/np.exp(1), ls = '--', color = 'k', 
                  alpha = 0.8, label = 'threshold')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].legend(loc = 'lower right')
    ax[i].set_title(j, fontsize=8)

fig.supylabel('Correlation Coefficient')
fig.supxlabel('Time Lags')
fig.suptitle('Auto Correlation Function')
plt.tight_layout()
plt.show()

# 2 x 4 Mutual Information

fig, ax = plt.subplots(1, 4, sharex = True, figsize = (16, 5))
ax = ax.flatten()

for i, j in enumerate(full_df.columns):
    
    mutual_info = tdmi.tdmi(full_df[j], 9, 4)
    ax[i].plot(np.arange(1,10,1), mutual_info, 'o-', )
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, weight = 'bold')
  
fig.supylabel('Mutual Information')
fig.supxlabel('Time Lags')
fig.suptitle('Average Mutual Information', weight = 'bold')
plt.tight_layout()
plt.savefig('fourcluster_mi.png', dpi = 600)
plt.show()

tau_all = [7, 3, 8, 6]

# 2 x 4 Cao's FNN

fig, ax = plt.subplots(1, 4, sharex = True, figsize = (16, 5))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(full_df.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(full_df[j], e, tau_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-')
    # ax[i].axhline(y = 0.9, ls = '--', label = 'threshold', color = 'grey')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, weight = 'bold')
    

fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension", weight = 'bold')
plt.tight_layout()
plt.savefig('fourcluster_caofnn.png', dpi = 600)
plt.show()

E_all = [6, 5, 5, 4]

# 2 x 4 shadow manifold

fig, ax = plt.subplots(1, 4, subplot_kw = dict(projection = '3d'), figsize = (16, 5))
ax = ax.flatten()

for i, j in enumerate(full_df.columns): 
    
    M = mv.build_shadow(full_df[j], E_all[i], tau_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, weight = 'bold')
    if i == 0:
        ax[i].set_xlabel('AMU(t)')
        ax[i].set_ylabel(r'AMU(t-$\tau$)')
        ax[i].set_zlabel(r'AMU(t-$\tau$)')
    else:
        None
plt.suptitle('Shadow Manifolds of Clusters of AMU in Animals', weight = 'bold')
plt.tight_layout()
plt.savefig('Shadow Manifolds of Clusters of AMU in Animals.png', dpi = 600)
plt.show()

r'''fig, ax = plt.subplots(2, 4, subplot_kw=dict(projection='3d'), figsize=(14, 7))
ax = ax.flatten()

manifolds = []
line_plots = []

for i, j in enumerate(single_df.columns): 
    M = mv.build_shadow(single_df[j], E_all[i], tau_all[i])
    manifolds.append(M)

    line, = ax[i].plot([], [], [], lw=1.2)
    line_plots.append(line)
    
    ax[i].set_xlim(M[:, 0].min(), M[:, 0].max())
    ax[i].set_ylim(M[:, 1].min(), M[:, 1].max())
    ax[i].set_zlim(M[:, 2].min(), M[:, 2].max())
    ax[i].set_title(j, size=9)
    
    ax[i].xaxis.set_tick_params(labelsize=7)
    ax[i].yaxis.set_tick_params(labelsize=7)
    ax[i].zaxis.set_tick_params(labelsize=7)

total_frames = 180 

def update(frame):

    azim_angle = (frame / total_frames) * 360
    elev_angle = 20 + 10 * np.sin(np.radians(azim_angle * 2))

    for i in range(len(ax)):
        M = manifolds[i]
        total_points = len(M)
        
        current_idx = int((frame / total_frames) * total_points)
        current_idx = max(2, current_idx) 
        
        x = M[:current_idx, 0]
        y = M[:current_idx, 1]
        z = M[:current_idx, 2]
        
        line_plots[i].set_data(x, y)
        line_plots[i].set_3d_properties(z)
        
        ax[i].view_init(elev=elev_angle, azim=azim_angle)
        
    'return line_plots

ani = FuncAnimation(
    fig, 
    update, 
    frames=total_frames, 
    interval=40, 
    blit=Fa'lse)

plt.tight_layout()

ani.save('evolving_manifold_rotation.gif', writer='pillow', fps=25)
plt.close(fig)'''

'''# 4 x 4 causal relationship
L_range = [np.arange(38, 94, 5)]
cols = single_df.columns
nvars = len(cols)

cols = single_df.columns
nvars = len(cols)

L_range = np.arange(38, 94, 5)

fig, axes = plt.subplots(
    nvars,
    nvars,
    figsize=(3*nvars, 3*nvars),
    sharex=True,
    sharey=True
)

for i, cause in enumerate(cols):

    for j, effect in enumerate(cols):

        ax = axes[i, j]

        # skip self-causation
        if i == j:
            ax.axis('off')
            continue

        ccm_result(
            CCM.CCM,
            single_df[cause].values,
            single_df[effect].values,
            L_range,

            tau_y=tau_all[j],
            tau_x=tau_all[i],

            E_y=E_all[j],
            E_x=E_all[i],

            L1=f'{cause} → {effect}',
            L2=f'{effect} → {cause}',
            N=100,
            ax=ax
        )

        if i == 0:
            ax.set_title(effect, fontsize=10)

        if j == 0:
            ax.set_ylabel(cause, fontsize=10)

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd

cols = single_df.columns
nvars = len(cols)

ccm_matrix = np.zeros((nvars, nvars))

L = max(L_range)

for i, cause in enumerate(cols):

    for j, effect in enumerate(cols):

        if i == j:
            ccm_matrix[i, j] = np.nan
            continue

        ccm = CCM.CCM(
            single_df[cause].values,
            single_df[effect].values,

            E_all[j],      # effect manifold
            tau_all[j],
            L
        )

        rho = ccm.causality()[0][1]

        ccm_matrix[i, j] = rho

ccm_df = pd.DataFrame(
    ccm_matrix,
    index=cols,
    columns=cols
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

sns.heatmap(
    ccm_df,
    cmap='RdBu_r',
    center=0,
    annot=True,
    fmt='.2f',
    linewidths=0.5
)

plt.title('CCM Skill Matrix')
plt.xlabel('Effect')
plt.ylabel('Cause')

plt.tight_layout()
plt.show()'''
# %% Inferring Causal Relationship - goes

titles = full_df.columns

# 1) Temperature

L_ranges = [np.arange(48, 90, 5),
           np.arange(23, 90, 5), np.arange(48, 90, 5),
           np.arange(28, 90, 5)]

figs = []
for i in range(4):
    fig_ax = ccm_result_1(CCM.CCM, full_df.iloc[:,i].values, c1_processed['Temperature'], L_ranges[i], 
                          tau_c1[0], tau_all[i], E_c1[0], E_all[i], 'AMU', 'Temperature')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 4, figsize=(16, 5))
ax = ax.flatten()

for i in range(4):

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

    ax[i].set_title(source_ax.get_title(loc='center'), weight = 'bold')
    ax[i].set_xlabel(source_ax.get_xlabel())
    ax[i].set_ylabel(source_ax.get_ylabel())
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
plt.suptitle('Convergent Cross Mapping Skill between Temperature and AMU cluster', weight = 'bold')
plt.tight_layout()
plt.savefig('CCM_Temp_4_clust.png', dpi = 600)
plt.show()

# 2. Relative Humidity

figs = []
for i in range(4):
    fig_ax = ccm_result_1(CCM.CCM, full_df.iloc[:,i].values, c1_processed['RH'], L_ranges[i], 
                          tau_c1[1], tau_all[i], E_c1[1], E_all[i], 'AMU', 'RH')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 4, figsize=(16, 5))
ax = ax.flatten()

for i in range(4):

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

    ax[i].set_title(source_ax.get_title(loc='center'), weight = 'bold')
    ax[i].set_xlabel(source_ax.get_xlabel())
    ax[i].set_ylabel(source_ax.get_ylabel())
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()

plt.suptitle('Convergent Cross Mapping Skill between Relative Humidity and AMU cluster', weight = 'bold')
plt.tight_layout()
plt.savefig('CCM_RH_4_clust.png', dpi = 600)
plt.show()

# 3. Wind Speed

figs = []
for i in range(4):
    fig_ax = ccm_result_1(CCM.CCM, full_df.iloc[:,i].values, c2_processed['Windspeed'], L_ranges[i], 
                          tau_c2[0], tau_all[i], E_c2[0], E_all[i], 'AMU', 'Windspeed')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 4, figsize=(16, 5))
ax = ax.flatten()

for i in range(4):

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

    ax[i].set_title(source_ax.get_title(loc='center'))
    ax[i].set_xlabel(source_ax.get_xlabel())
    ax[i].set_ylabel(source_ax.get_ylabel())
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
        
plt.suptitle('Convergent Cross Mapping Skill between Windspeed and AMU cluster', weight = 'bold')
plt.tight_layout()
plt.savefig('CCM_wind_clust.png', dpi = 600)
plt.show()

# 4. Precipitation

figs = []
for i in range(4):
    fig_ax = ccm_result_1(CCM.CCM, full_df.iloc[:,i].values, c2_processed['Precipitation'], L_ranges[i], 
                          tau_c2[1], tau_all[i], E_c2[1], E_all[i], 'AMU', 'Precipitation')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(1, 4, figsize=(16, 5))
ax = ax.flatten()

for i in range(4):

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

    ax[i].set_title(source_ax.get_title(loc='center'))
    ax[i].set_xlabel(source_ax.get_xlabel())
    ax[i].set_ylabel(source_ax.get_ylabel())
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
plt.suptitle('Convergent Cross Mapping Skill between Precipitation and AMU cluster', weight = 'bold')
plt.tight_layout()
plt.savefig('CCM_Prec_clust.png', dpi = 600)
plt.show()
