# %% Author Note

'''

This code is written by Prem Kumar Loganathan for the MSc Thesis titled, 
'Investigating Climate-Driven Causal Relationships in Antimicrobial Use and Resistance in Animals'

All the class objects and user-defined functions needed for this code is made available in the same repository. 
No explanation will be provided anywhere in code until and unless necessary.

'''

#  %% Import Libraries

import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# %% Other user-defined functions

def has_consecutive_nans(series, threshold): # Thanks to GenAI
    
    nan_vals = series.isna()
    group = (~nan_vals).cumsum()
    streaks = nan_vals.groupby(group).cumsum()
    
    return (streaks >= threshold).any()

def save_ccm_plot(img, title, filename):
    fig, ax = plt.subplots(figsize=(6,6), sharex=True)

    for line in img.lines:
        ax.plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            c=line.get_color(),
            ls=line.get_linestyle(),
            lw=line.get_linewidth(),
            marker=line.get_marker()
        )

    ax.set_title(title, size=11, weight='bold')
    ax.set_xlabel('Library Size [L]')
    ax.set_ylabel('Cross Map Skill (ρ)')
    ax.legend(loc='best')
    ax.grid('--', c='grey', alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()

# %% Loading Dataset

# 1) AMU Animals
amu_data_path = r'D:\Education\M.Sc\Thesis\Codes\New folder\AMU_DF.csv'
amu_df = pd.read_csv(amu_data_path)

# 1) AMR Animals
amr_data_path = r'D:\Education\M.Sc\Thesis\Codes\New folder\EFSA data Ecoli.xlsx'
amr_df = pd.read_excel(amr_data_path, engine = 'calamine')

# 2) Temperature
temp_path = r'D:\Education\M.Sc\Thesis\Codes\New folder\BE_Temp_df.csv'
full_temp = pd.read_csv(temp_path)
temp = full_temp.iloc[2:,:2]
temp.index = temp['YY-MM']
temp = temp.drop(columns = 'YY-MM')

# 3) Relative Humidity
rh_path = r'D:\Education\M.Sc\Thesis\Codes\New folder\BE_Humi_df.csv'
rh = pd.read_csv(rh_path)
rh = rh['Mean_Hum'][2:]

# 4) Precipitation
prec_path = r'D:\Education\M.Sc\Thesis\Codes\New folder\BE_Prec_df.csv'
full_prec = pd.read_csv(prec_path)
prec = full_prec['Mean_Prec'][2:]

# 5) Windspeed
ws_path = r'D:\Education\M.Sc\Thesis\Codes\New folder\BE_Wind_df.csv'
full_ws = pd.read_csv(ws_path)
ws = full_ws['Mean_Wind'][2:]

# %% Preprocessing AMU Data

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

animal_name_map = {'PIG': 'Pigs', 'VECLF': 'Calves', 'PLTR': 'Poultry'}

amu_df['Active_Substance'] = amu_df['Active_Substance'].replace(ab_name)
amu_df['AnimalType'] = amu_df['AnimalType'].replace(animal_name_map)

amu_df_for_clustering = amu_df.groupby(['YY-MM', 'Active_Substance', 'AnimalType'])[['Total_Active_Substance']].sum().reset_index()
table_for_clustering = pd.pivot_table(data = amu_df_for_clustering, index = 'YY-MM', values = 'Total_Active_Substance', columns = ['AnimalType', 'Active_Substance'])

cols_to_drop = [col for col in table_for_clustering.columns if has_consecutive_nans(table_for_clustering[col], 5)]
amu_df_cleaned = table_for_clustering.drop(columns = cols_to_drop)
loss = ((table_for_clustering.shape[1] - amu_df_cleaned.shape[1]) / table_for_clustering.shape[1]) * 100
print(f'Loss: {np.round(loss, 2)}%')

columns_in_amu = [( 'Calves',               'Ampicillin'), ( 'Calves',              'Doxycycline'), ( 'Calves',          'Oxytetracycline'), ( 'Calves', 'Trimethoprim_Sulfonamide'),
(   'Pigs',               'Ampicillin'), (   'Pigs',        'Chlortetracycline'), (   'Pigs',              'Doxycycline'), (   'Pigs',          'Oxytetracycline'), (   'Pigs', 'Trimethoprim_Sulfonamide'),
('Poultry',              'Doxycycline'), ('Poultry', 'Trimethoprim_Sulfonamide')]
amu_new = amu_df_cleaned[columns_in_amu].copy()

amu_new.columns = pd.MultiIndex.from_tuples([(animal, 'Tetracycline' if ab in ['Doxycycline', 'Oxytetracycline', 'Chlortetracycline'] else ab) for animal, ab in amu_new.columns])
amu_new_grouped = amu_new.groupby(level = [0, 1], axis = 1).mean()
amu_new_processed = preprocessing(amu_new_grouped.interpolate().rolling(window = 5, center = True).mean().dropna())

# %% Preprocessing AMR Data

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

animal_ab_table = pd.pivot_table(data = gb_animal_res, index = 'YY-MM', 
                                 columns = ['Animal Type', 'Active Substance'], values = 'Resistance (%)')
animal_ab_table = animal_ab_table.reindex(period_range)
animal_count_table = pd.pivot_table(data = gb_animal_res, index = 'YY-MM', 
                                 columns = ['Animal Type', 'Active Substance'], values = 'tested')
animal_count_table = animal_count_table.reindex(period_range)
animals = sorted(gb_animal_res['Animal Type'].unique())

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

to_exclude = [
    (animal, abx)
    for animal, abx_list in exclusion_map.items()
    for abx in abx_list]

animal_ab_df_cleaned_new = animal_ab_df_cleaned.loc[:, ~animal_ab_df_cleaned.columns.isin(to_exclude)]
animal_ab_df_cleaned_filled = animal_ab_df_cleaned_new.interpolate()
animal_ab_df_cleaned_filled_smoothed = (animal_ab_df_cleaned_filled.rolling(window=5, center=True).mean().dropna())

columns_in_amr = [( 'Calves',       'Ampicillin'), ( 'Calves',     'Tetracycline'), ( 'Calves', 'Sulfamethoxazole'),  ( 'Calves',     'Trimethoprim'),
(   'Pigs',       'Ampicillin'), (   'Pigs',     'Tetracycline'), (   'Pigs', 'Sulfamethoxazole'),  (   'Pigs',     'Trimethoprim'),
('Poultry',     'Tetracycline'), ('Poultry', 'Sulfamethoxazole'),('Poultry',     'Trimethoprim')]
amr_new_processed = preprocessing(animal_ab_df_cleaned_filled_smoothed[columns_in_amr])

# %% AMU x AMR causal inference

L_ranges = [np.arange(38, 93, 5), np.arange(43, 93, 5), np.arange(48, 93, 5), np.arange(43, 93, 5),
        np.arange(43, 93, 5), np.arange(33, 93, 5), np.arange(43, 93, 5), np.arange(43, 93, 5),
        np.arange(38, 93, 5), np.arange(48, 93, 5), np.arange(48, 93, 5)]

############################################------CALVES--------#####################################

# 1) Calves - Ampicillin
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,0], amr_new_processed.iloc[:,0], L_ranges[0],
                   tau_y=8, tau_x=6, E_y=4, E_x=4, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Calves - Ampicillin',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Calves_Ampicillin.png')

# 2) Calves - Tetracycline
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,1], amr_new_processed.iloc[:,1], L_ranges[1],
                   tau_y=5, tau_x=6, E_y=7, E_x=4, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Calves - Tetracycline',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Calves_Tetracycline.png')

# 3) Calves - Sulfamethoxazole
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,2], amr_new_processed.iloc[:,2], L_ranges[2],
                   tau_y=8, tau_x=6, E_y=5, E_x=6, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Calves - Sulfamethoxazole',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Calves_Sulfamethoxazole.png')

# 4) Calves - Trimethoprim
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,2], amr_new_processed.iloc[:,3], L_ranges[3],
                   tau_y=7, tau_x=6, E_y=4, E_x=6, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Calves - Trimethoprim',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Calves_Trimethoprim.png')

############################################------PIGS--------#####################################
# 5) Pigs - Ampicillin
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,3], amr_new_processed.iloc[:,4], L_ranges[4],
                   tau_y=6, tau_x=6, E_y=6, E_x=5, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Pigs - Ampicillin',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Pigs_Ampicillin.png')

# 6) Pigs - Tetracycline
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,4], amr_new_processed.iloc[:,5], L_ranges[5],
                   tau_y=7, tau_x=6, E_y=4, E_x=4, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Pigs - Tetracycline',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Pigs_Tetracycline.png')

# 7) Pigs - Sulfamethoxazole
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,5], amr_new_processed.iloc[:,6], L_ranges[6],
                   tau_y=6, tau_x=6, E_y=6, E_x=6, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Pigs - Sulfamethoxazole',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Pigs_Sulfamethoxazole.png')

# 8) Pigs - Trimethoprim
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,5], amr_new_processed.iloc[:,7], L_ranges[7],
                   tau_y=6, tau_x=6, E_y=4, E_x=6, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Pigs - Trimethoprim',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Pigs_Trimethoprim.png')
# Check cross-mapping
ccm_XY = CCM.CCM(amu_new_processed.iloc[:,5], amr_new_processed.iloc[:,7], 4, 6, 83)
ccm_XY.causality(tp = 0)[0][1]
ccm_XY.plot_ccm_correlation()

############################################------POULTRY--------#####################################
# 9) Poultry - Tetracycline
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,6], amr_new_processed.iloc[:,8], L_ranges[8],
                   tau_y=6, tau_x=5, E_y=4, E_x=6, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Poultry - Tetracycline',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Poultry_Tetracycline.png')

# 10) Poultry - Sulfamethoxazole
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,7], amr_new_processed.iloc[:,9], L_ranges[9],
                   tau_y=5, tau_x=8, E_y=5, E_x=5, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Poultry - Sulfamethoxazole',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Poultry_Sulfamethoxazole.png')
# Check cross-mapping
ccm_YX = CCM.CCM(amr_new_processed.iloc[:,9], amu_new_processed.iloc[:,7], 5, 8, 83)
ccm_YX.causality(tp = 0)[0][1]
ccm_YX.plot_ccm_correlation()

# 11) Poultry - Trimethoprim
img = ccm_result_1(CCM.CCM, amu_new_processed.iloc[:,7], amr_new_processed.iloc[:,10], L_ranges[10],
                   tau_y=6, tau_x=8, E_y=5, E_x=5, variable1='AMU', variable2='AMR')
save_ccm_plot(img, 'Poultry - Trimethoprim',
              r'D:\Education\M.Sc\Thesis\Codes\New folder\Thesis Images\Poultry_Trimethoprim.png')
