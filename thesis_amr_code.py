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

# Grouped in terms of Animal Type, and Active Substance

columns_of_interest = ['labIsolCode', 'matrix', 'sampY', 'sampM', 'Active Substance', 'MIC', 'cutoffValue']
amr_data = amr_df[columns_of_interest]
amr_data = amr_data[amr_data['Active Substance'] != 'Amikacin']

matrix_map = {'PRI 035': 'Pigs', 'PRI 036': 'Calves', 'PRI 019 Broilers': 'Poultry', 'PRI 019 Turkeys': 'Poultry'}
amr_data['Animal Type'] = amr_data['matrix'].map(matrix_map)
amr_data['Positive'] = (amr_data['MIC'] > amr_data['cutoffValue']).astype(int)
amr_data['YY-MM'] = amr_data['sampY'].astype(str) + '-' + amr_data['sampM'].astype(str)
amr_data['YY-MM'] = pd.to_datetime(amr_data['YY-MM'], format = '%Y-%m')
period_range = pd.date_range(start = amr_data['YY-MM'].min(),
                             end = amr_data['YY-MM'].max(), freq = 'MS')

# gb_animal = ['YY-MM', 'Animal Type','labIsolCode', 'Active Substance', 'Positive']
# gb_animal_table = amr_data[gb_animal]
# gb_animal_res = gb_animal_table.groupby(['YY-MM', 'Animal Type', 'Active Substance'])['Positive'].agg(resistance = 'sum', tested = 'count').reset_index()
# gb_animal_res['Resistance (%)'] = (gb_animal_res['resistance'] / gb_animal_res['tested'])
# gb_animal_mdr = gb_animal_table.groupby(['YY-MM', 'labIsolCode', 'Animal Type'])['Positive'].sum().reset_index(name = 'No of Positives')


# Grouped only if Active Substance

gb_as = ['YY-MM', 'labIsolCode', 'Active Substance', 'Positive']
gb_as_table = amr_data[gb_as]

gb_as_res = gb_as_table.groupby(['YY-MM', 'Active Substance'])['Positive'].agg(resistance = 'sum', tested = 'count').reset_index()
gb_as_res['Resistance (%)'] = (gb_as_res['resistance'] / gb_as_res['tested']) * 100

gb_as_res_mdr = gb_as_table.groupby(['YY-MM', 'labIsolCode'])['Positive'].sum().reset_index(name = 'No of Positives')
gb_as_res_mdr['atleast_one'] = (gb_as_res_mdr['No of Positives'] >= 1).astype(int)
gb_as_res_mdr['atleast_two'] = (gb_as_res_mdr['No of Positives'] >= 2).astype(int)
gb_as_res_mdr['atleast_three'] = (gb_as_res_mdr['No of Positives'] >= 3).astype(int)

gb_as_res_mdr_res = gb_as_res_mdr.groupby('YY-MM').agg(isolates = ('labIsolCode', 'nunique'),
            total_atleast_one = ('atleast_one', 'sum'), total_atleast_two = ('atleast_two', 'sum'), 
           total_atleast_three = ('atleast_three', 'sum')).reset_index()

gb_as_res_mdr_res['atleast_one (%)'] = gb_as_res_mdr_res['total_atleast_one'] / gb_as_res_mdr_res['isolates'] * 100
gb_as_res_mdr_res['atleast_two (%)'] = gb_as_res_mdr_res['total_atleast_two'] / gb_as_res_mdr_res['isolates'] * 100
gb_as_res_mdr_res['atleast_three (%)'] = gb_as_res_mdr_res['total_atleast_three'] / gb_as_res_mdr_res['isolates'] * 100
gb_as_res_mdr_res = gb_as_res_mdr_res.set_index('YY-MM').reindex(period_range).reset_index().rename(columns = {'index' : 'YY-MM'})

antibiotics_table = pd.pivot_table(data = gb_as_res, values = 'Resistance (%)', index = 'YY-MM', columns = 'Active Substance')
antibiotics_table = antibiotics_table.reindex(period_range)

# All 3 visulaisation plot

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(gb_as_res_mdr_res['YY-MM'], gb_as_res_mdr_res['atleast_one (%)'], 's-', label='MDR > 1')
ax.plot(gb_as_res_mdr_res['YY-MM'], gb_as_res_mdr_res['atleast_two (%)'], 's-', label='MDR > 2')
ax.plot(gb_as_res_mdr_res['YY-MM'], gb_as_res_mdr_res['atleast_three (%)'], '^-', label='MDR > 3')

ax.set_xmargin(0)
ax.tick_params('x', rotation = 90)
ax.set_ylabel('Resistant Percentage (%)')
ax.set_xlabel('Time[Month]')
ax.set_title('AMR trend in Food Producing Animals in Belgium (2017-2024)')
ax.legend()
ax.grid('--', color = 'grey', alpha=0.3)
plt.tight_layout()
plt.show()

def has_consecutive_nans(series, threshold): # Thanks to GenAI
    
    nan_vals = series.isna()
    group = (~nan_vals).cumsum()
    streaks = nan_vals.groupby(group).cumsum()
    
    return (streaks >= threshold).any()

cols_to_drop = [col for col in antibiotics_table.columns if has_consecutive_nans(antibiotics_table[col], 5)]
amr_df_cleaned = antibiotics_table.drop(columns = cols_to_drop)
loss = ((antibiotics_table.shape[1] - amr_df_cleaned.shape[1]) / antibiotics_table.shape[1]) * 100
print(f'Loss: {np.round(loss, 2)}%')

amr_df_cleaned['Atleast One'] = gb_as_res_mdr_res['atleast_one (%)'].values
amr_df_cleaned['Atleast Two'] = gb_as_res_mdr_res['atleast_two (%)'].values
amr_df_cleaned['Atleast Three'] = gb_as_res_mdr_res['atleast_three (%)'].values

fig, ax = plt.subplots(3, 6, figsize=(16, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(amr_df_cleaned.columns):
    ax[i].plot(amr_df_cleaned.index, amr_df_cleaned[j], 'o-', label = j, markersize = 2)
    ax[i].set_title(j, loc = 'left', size = 8)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_xticks(amr_df_cleaned.index[::12])
    ax[i].tick_params('x', rotation = 90)

ABs_with_100_and_0_resistance = ['Meropeneme', 'Colistin', 'Tigecycline'] 
amr_df_cleaned_2 = amr_df_cleaned.drop(columns = ABs_with_100_and_0_resistance)

amr_df_filled = preprocessing(amr_df_cleaned_2.interpolate())
fig, ax = plt.subplots(3, 5, figsize=(16, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(amr_df_filled.columns):
    ax[i].plot(amr_df_filled.index, amr_df_filled[j], 'o-', label = j, markersize = 2)
    ax[i].set_title(j, loc = 'left', size = 8)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_xticks(amr_df_cleaned.index[::12])
    ax[i].tick_params('x', rotation = 90)

fig, ax = plt.subplots(3, 5, figsize = (16, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(amr_df_filled.columns):
    stl = STL(amr_df_filled[j], period = 13)
    res = stl.fit()
    ax[i].plot(res.trend)
    ax[i].set_title(f'{j} - Trend', loc = 'left', size = 8)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].tick_params('x', rotation = 90)
    
fig, ax = plt.subplots(3, 5, figsize = (16, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(amr_df_filled.columns):
    stl = STL(amr_df_filled[j], period = 13)
    res = stl.fit()
    ax[i].plot(res.seasonal)
    ax[i].set_title(f'{j} - Seasonal Pattern', loc = 'left', size = 8)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].tick_params('x', rotation = 90)
    
# Smoothening

fig, ax = plt.subplots(3, 5, figsize=(16, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(amr_df_filled.columns):
    ax[i].plot(amr_df_filled.index, amr_df_filled[j], 'o-', label = 'Original', markersize = 2)
    ax[i].plot(amr_df_filled.index, amr_df_filled[j].rolling(window = 3, center = True).mean(), 'o-', color = 'orange', label = 'Window = 3', markersize = 2)
    ax[i].plot(amr_df_filled.index, amr_df_filled[j].rolling(window = 5, center = True).mean(), 'o-', color = 'red', label = 'Window = 5', markersize = 2, alpha = 0.7) 
    ax[i].legend() if i == 0 else None
    ax[i].set_title(j, loc = 'left', size = 8)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_xticks(amr_df_cleaned.index[::12])
    ax[i].tick_params('x', rotation = 90)

# %% Clustering

amr_df_processed_one = amr_df_filled.rolling(window = 3, center = True).mean().dropna()
amr_df_processed_one = amr_df_processed_one.drop(columns = ['Atleast Two', 'Atleast Three'])
k_vals = np.arange(2, 11, 1)
sse = []
shilloutte = []
centers = []
labels = []

for i in k_vals:
    kmeans = KMeans(n_clusters = i, random_state = 42, n_init = 20)
    kmeans.fit(amr_df_processed_one.T)
    labels.append(kmeans.labels_)
    centers.append(kmeans.cluster_centers_)
    sse.append(kmeans.inertia_)
    shilloutte.append(silhouette_score(amr_df_processed_one.T, kmeans.labels_))

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
plt.savefig('Choosing Optimal Clusters for AMU.png', dpi = 600)
plt.show()

c = KMeans(n_clusters = 7, random_state = 42, n_init = 20).fit(amr_df_processed_one.T)
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
        
coordinates_plot(amr_df_processed_one.T.values, c_l, )

c_l = np.array(c_l)
cols = amr_df_processed_one.columns
colors = cm.tab10.colors

fig, ax = plt.subplots(7, 1, figsize = (10, 12), sharex = True)
ax = ax.flatten()
cluster_means = {}
for i, clust in enumerate(np.unique(c_l)):
    
    selected_cols = cols[c_l == clust]
    data = amr_df_processed_one[selected_cols.values]
    cluster_means[f'clust_{i}'] = np.mean(data, axis=1)

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
fig.suptitle('Antimicrobial Resistance in Food Producing Animals')
plt.tight_layout()
plt.show()

tsne = TSNE(n_components = 2, perplexity = 6, random_state=42)
X_new = tsne.fit_transform(amr_df_processed_one.T)
print(f'KL divergence: {tsne.kl_divergence_:.4f}')

X_tsne = pd.DataFrame(X_new, columns=['TSNE1', 'TSNE2'], index=cols)
X_tsne['Cluster'] = c_l

fig, ax = plt.subplots(figsize=(10, 8))

for cluster in sorted(X_tsne['Cluster'].unique()):

    data = X_tsne[X_tsne['Cluster'] == cluster]
    ax.scatter(data['TSNE1'], data['TSNE2'], s=25,
        color=colors[int(cluster) % len(colors)], label=f'Cluster {cluster}')
    for idx, row in data.iterrows():
        ax.text(row['TSNE1'] + 0.1, row['TSNE2'] + 0.1, idx, fontsize=8.5)

ax.set_title('t-SNE projection of AMU in Food Producing \nAnimals in Belgium coloured by K-means clusters', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.grid(alpha=0.3)
ax.legend(title='Cluster')

# %% Climate Variables Smoothing

Temperature = pd.DataFrame(temp.values).rolling(window = 5, center = True).mean().dropna()
Windspeed = pd.DataFrame(ws.values).rolling(window = 3, center = True).mean().dropna()
RH = pd.DataFrame(rh.values).rolling(window = 5, center = True).mean().dropna()
Precipitation = pd.DataFrame(prec.values).rolling(window = 3, center = True).mean().dropna()

# %% Checking Convergent Cross Mapping for 14 ABs + 3 MDR

#df for testing
amr_df_processed_one = amr_df_filled.rolling(window = 3, center = True).mean().dropna()

fig, ax = plt.subplots(3, 5, sharex = True, figsize = (12, 8))
ax = ax.flatten()

for i, j in enumerate(amr_df_processed_one.columns):
    data = amr_df_processed_one[j]
    mutual_info = tdmi.tdmi(data, 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_title(j, loc = 'left', size = 9.5)

fig.supxlabel('Time [Months]')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

tau_all = [3, 3, 4, 5, 3, 4, 3, 4, 5, 4, 5, 5, 4, 4]

fig, ax = plt.subplots(3, 5, sharex = True, figsize = (12, 6))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(amr_df_processed_one.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(amr_df_processed_one[j], e, tau_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)
    
fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

E_all = [5, 6, 5, 4, 4, 4, 5, 7, 7, 5, 5, 4, 5, 4, 4]

fig, ax = plt.subplots(3, 5, subplot_kw = dict(projection = '3d'), figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(amr_df_processed_one.columns): 
    
    M = mv.build_shadow(amr_df_processed_one[j], E_all[i], tau_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, size = 9)

titles = amr_df_processed_one.columns

# 1) Temperature

L_ranges = [np.arange(28, 90, 5)] * 7 + [np.arange(38, 90, 5)] + [np.arange(43, 90, 5)] + [np.arange(28, 90, 2)] + [np.arange(33, 90, 2)] + [np.arange(28 , 90, 5)] * 3

fig, ax = plt.subplots(3, 5, figsize = (18, 8))
ax = ax.flatten()

figs = []
for i in range(14):
    fig_ax = ccm_result_1(CCM.CCM, amr_df_processed_one.iloc[1:-1,i].values, Temperature.values.ravel(), L_ranges[i], 
                          4, tau_all[i], 4, E_all[i])
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(3, 5, figsize=(20, 8))
ax = ax.flatten()

for i in range(14):

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

plt.tight_layout()
plt.show()