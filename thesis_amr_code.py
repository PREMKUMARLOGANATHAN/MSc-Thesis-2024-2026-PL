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
import pyEDM
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

# %% AMR Data preprocessing pt.1

# Grouped in terms of Animal Type, and Active Substance

columns_of_interest = ['labIsolCode', 'matrix', 'sampY', 'sampM', 'Active Substance', 'MIC', 'cutoffValue']
amr_data = amr_df[columns_of_interest]
# amr_data = amr_data[amr_data['Active Substance'] != 'Amikacin']

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

fig, ax = plt.subplots(3, 6, figsize=(20, 10), sharex = True)
ax = ax.flatten()

for i, j in enumerate(amr_df_cleaned.columns):
    if j == 'Meropeneme' or j == 'Colistin' or j == 'Tigecycline':
        line_color = 'red'
        title_color = 'red'
    else: 
        line_color = 'blue'
        title_color = 'black'
    ax[i].plot(amr_df_cleaned.index, amr_df_cleaned[j], 'o-', color = line_color, label = j, markersize = 2)
    ax[i].set_title(j, size = 9, color = title_color)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_xticks(amr_df_cleaned.index[::12])
    ax[i].tick_params('x', rotation = 90)

ax[11].tick_params(axis='x', labelbottom=True)
fig.delaxes(ax[17])
fig.supxlabel('Time [Months]')
fig.supylabel('Resistance (%)')
fig.suptitle('E coli Resistance Occurance in Food Producing Animals in Belgium')
plt.tight_layout()
plt.show()

ABs_with_100_and_0_resistance = ['Meropeneme', 'Colistin', 'Tigecycline'] 
amr_df_cleaned_2 = amr_df_cleaned.drop(columns = ABs_with_100_and_0_resistance)

amr_df_filled = amr_df_cleaned_2.interpolate()
fig, ax = plt.subplots(3, 5, figsize=(20, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(amr_df_filled.columns):
    ax[i].plot(amr_df_filled.index, amr_df_filled[j], 'o-', label = j, markersize = 2)
    ax[i].set_title(j, loc = 'left', size = 8)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_xticks(amr_df_cleaned.index[::12])
    ax[i].tick_params('x', rotation = 90)

fig, ax = plt.subplots(3, 5, figsize = (20, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(amr_df_filled.columns):
    stl = STL(amr_df_filled[j], period = 13)
    res = stl.fit()
    ax[i].plot(res.trend)
    ax[i].set_title(f'{j} - Trend', loc = 'left', size = 8)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].tick_params('x', rotation = 90)
    
fig, ax = plt.subplots(3, 5, figsize = (20, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(amr_df_filled.columns):
    stl = STL(amr_df_filled[j], period = 13)
    res = stl.fit()
    ax[i].plot(res.seasonal)
    ax[i].set_title(f'{j} - Seasonal Pattern', loc = 'left', size = 8)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].tick_params('x', rotation = 90)
    
# Smoothening

fig, ax = plt.subplots(3, 5, figsize=(20, 12), sharex = True)
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

amr_df_processed_one = preprocessing(amr_df_cleaned_2.interpolate().rolling(window = 5, center = True).mean().dropna())
amr_df_processed_one_c = amr_df_processed_one.drop(columns = ['Atleast One', 'Atleast Two', 'Atleast Three'])
k_vals = np.arange(2, 11, 1)
sse = []
shilloutte = []
centers = []
labels = []

for i in k_vals:
    kmeans = KMeans(n_clusters = i, random_state = 42, n_init = 20)
    kmeans.fit(amr_df_processed_one_c.T)
    labels.append(kmeans.labels_)
    centers.append(kmeans.cluster_centers_)
    sse.append(kmeans.inertia_)
    shilloutte.append(silhouette_score(amr_df_processed_one_c.T, kmeans.labels_))

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

c = KMeans(n_clusters = 6, random_state = 42, n_init = 20).fit(amr_df_processed_one_c.T)
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
        
coordinates_plot(amr_df_processed_one_c.T.values, c_l, )

c_l = np.array(c_l)
cols = amr_df_processed_one_c.columns
colors = cm.tab10.colors

fig, ax = plt.subplots(6, 1, figsize = (10, 12), sharex = True)
ax = ax.flatten()
cluster_means = {}
for i, clust in enumerate(np.unique(c_l)):
    
    selected_cols = cols[c_l == clust]
    data = amr_df_processed_one_c[selected_cols.values]
    cluster_means[f'clust_{i +1}'] = np.mean(data, axis=1)

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
X_new = tsne.fit_transform(amr_df_processed_one_c.T)
print(f'KL divergence: {tsne.kl_divergence_:.4f}')

X_tsne = pd.DataFrame(X_new, columns=['TSNE1', 'TSNE2'], index=cols)
X_tsne['Cluster'] = c_l

fig, ax = plt.subplots(figsize=(10, 8))

for cluster in sorted(X_tsne['Cluster'].unique()):

    data = X_tsne[X_tsne['Cluster'] == cluster]
    ax.scatter(data['TSNE1'], data['TSNE2'], s=25,
        color=colors[int(cluster) % len(colors)], label=f'Cluster {cluster +1}')
    for idx, row in data.iterrows():
        ax.text(row['TSNE1'] + 0.1, row['TSNE2'] + 0.1, idx, fontsize=8.5)

ax.set_title('t-SNE projection of AMU in Food Producing \nAnimals in Belgium coloured by K-means clusters', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.grid(alpha=0.3)
ax.legend(title='Cluster')

# %% Climate Variable's Smoothing

Temperature = preprocessing(temp.rolling(window = 5, center = True).mean().dropna())
Windspeed = preprocessing(ws.rolling(window = 3, center = True).mean().dropna())
RH = preprocessing(rh.rolling(window = 5, center = True).mean().dropna())
Precipitation = preprocessing(prec.rolling(window = 3, center = True).mean().dropna())

# %% Checking Convergent Cross Mapping for 14 ABs + 3 MDR

#df for testing
amr_df_processed_one = preprocessing(amr_df_cleaned_2.interpolate().rolling(window = 5, center = True).mean().dropna())

fig, ax = plt.subplots(3, 5, sharex = True, figsize = (20, 10))
ax = ax.flatten()

for i, j in enumerate(amr_df_processed_one.columns):
    data = amr_df_processed_one[j]
    mutual_info = tdmi.tdmi(data, 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_title(j, loc = 'left', size = 9.5)

fig.delaxes(ax[len(amr_df_processed_one.columns)])
fig.supxlabel('Time [Months]')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

tau_all = [7, 4, 4, 4, 6, 4, 6, 5, 5, 7, 6, 7, 7, 6]

fig, ax = plt.subplots(3, 5, sharex = True, figsize = (14, 8))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(amr_df_processed_one.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(amr_df_processed_one[j], e, tau_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)

fig.delaxes(ax[len(amr_df_processed_one.columns)])
fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

E_all = [5, 8, 7, 7, 5, 6, 6, 6, 5, 4, 5, 7, 5, 4]

fig, ax = plt.subplots(3, 5, subplot_kw = dict(projection = '3d'), figsize = (15, 8))
ax = ax.flatten()

for i, j in enumerate(amr_df_processed_one.columns): 
    
    M = mv.build_shadow(amr_df_processed_one[j], E_all[i], tau_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, loc = 'left', size = 9)
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)
    
fig.delaxes(ax[len(amr_df_processed_one.columns)])
plt.tight_layout()
plt.show()

titles = amr_df_processed_one.columns

# 1) Temperature

L_ranges = [np.arange(43, 90, 5), np.arange(38, 90, 5),
            np.arange(33, 90, 5), np.arange(33, 90, 5), 
            np.arange(38, 90, 5), np.arange(28, 90, 5), 
            np.arange(43, 90, 5), np.arange(38, 90, 5), 
            np.arange(33, 90, 5), np.arange(33, 90, 5), 
            np.arange(38, 90, 5), np.arange(53, 90, 5), 
            np.arange(43, 90, 5), np.arange(28, 90, 5)]

figs = []
for i in range(14):
    fig_ax = ccm_result_1(CCM.CCM, amr_df_processed_one.iloc[:,i].values, Temperature.values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'Temperature')
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

# 2) Relative Humidity

figs = []
for i in range(14):
    fig_ax = ccm_result_1(CCM.CCM, amr_df_processed_one.iloc[:,i].values, RH.values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'RH')
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

# 2) Wind Speed

figs = []
for i in range(14):
    fig_ax = ccm_result_1(CCM.CCM, amr_df_processed_one.iloc[:,i].values, Windspeed[1:-1].values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'Windspeed')
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

# 4) Precipitation

figs = []
for i in range(14):
    fig_ax = ccm_result_1(CCM.CCM, amr_df_processed_one.iloc[:,i].values, Precipitation[1:-1].values, L_ranges[i], 
                          3, tau_all[i], 4, E_all[i], variable = 'Precipitation')
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

# %% CCM at w = 5

fig, ax = plt.subplots(3, 5, sharex = True, figsize = (14, 8))
ax = ax.flatten()

for i, j in enumerate(amr_df_processed_one.columns):
    data = amr_df_processed_one[j]
    mutual_info = tdmi.tdmi(data, 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_title(j, loc = 'left', size = 9.5)

fig.delaxes(ax[len(amr_df_processed_one.columns)])
fig.supxlabel('Time [Months]')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

tau_all = [4, 4, 4, 4, 4, 5, 6, 5, 6, 6, 5, 6, 8, 6]

fig, ax = plt.subplots(3, 5, sharex = True, figsize = (14, 8))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(amr_df_processed_one.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(amr_df_processed_one[j], e, tau_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)

fig.delaxes(ax[len(amr_df_processed_one.columns)])
fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

E_all = [5, 4, 4, 7, 6, 5, 6, 4, 5, 4, 5, 5, 3, 4]

fig, ax = plt.subplots(3, 5, subplot_kw = dict(projection = '3d'), figsize = (15, 8))
ax = ax.flatten()

for i, j in enumerate(amr_df_processed_one.columns): 
    
    M = mv.build_shadow(amr_df_processed_one[j], E_all[i], tau_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, loc = 'left', size = 9)
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)
    
fig.delaxes(ax[len(amr_df_processed_one.columns)])
plt.tight_layout()
plt.show()

titles = amr_df_processed_one.columns

# 1) Temperature

L_ranges = [np.arange(28, 90, 5), np.arange(23, 90, 5),
            np.arange(23, 90, 5), np.arange(33, 90, 5), 
            np.arange(28, 90, 5), np.arange(38, 90, 5), 
            np.arange(38, 90, 5), np.arange(28, 90, 5), 
            np.arange(33, 90, 5), np.arange(28, 90, 5), 
            np.arange(33, 90, 5), np.arange(38, 90, 5), 
            np.arange(38, 90, 5), np.arange(28, 90, 5), ]

fig, ax = plt.subplots(3, 5, figsize = (18, 8))
ax = ax.flatten()

figs = []
for i in range(14):
    fig_ax = ccm_result_1(CCM.CCM, amr_df_processed_one.iloc[1:-1,i].values, Temperature.values, L_ranges[i], 
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

# 2) Relative Humidity

fig, ax = plt.subplots(3, 5, figsize = (18, 8))
ax = ax.flatten()

figs = []
for i in range(14):
    fig_ax = ccm_result_1(CCM.CCM, amr_df_processed_one.iloc[1:-1,i].values, RH.values, L_ranges[i], 
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

# 3) Wind Speed

fig, ax = plt.subplots(3, 5, figsize = (18, 8))
ax = ax.flatten()

figs = []
for i in range(14):
    fig_ax = ccm_result_1(CCM.CCM, amr_df_processed_one.iloc[:,i].values, Windspeed[1:-1].values, L_ranges[i], 
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

# 4) Precipitation

fig, ax = plt.subplots(3, 5, figsize = (18, 8))
ax = ax.flatten()

figs = []
for i in range(14):
    fig_ax = ccm_result_1(CCM.CCM, amr_df_processed_one.iloc[:,i].values, Precipitation[1:-1].values, L_ranges[i], 
                          3, tau_all[i], 4, E_all[i])
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

# %% CCM for 6 clusters

cluster_df = pd.DataFrame(cluster_means)
fig, ax = plt.subplots(2, 3, sharex = True, figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(cluster_df.columns):
    ax[i].plot(cluster_df.index, cluster_df[j], lw = 2)
    ax[i].set_title(j, loc = 'left', size = 9)
    ax[i].tick_params('x', rotation = 90)
    ax[i].grid('--', c = 'grey', alpha = 0.4)
    
fig.supxlabel('Time [Months]')
fig.supylabel('Resistance (Normalised)')
plt.tight_layout()
plt.show()

# Parameters

fig, ax = plt.subplots(2, 3, sharex = True, figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(cluster_df.columns):
    mutual_info = tdmi.tdmi(cluster_df[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', c = 'grey', alpha = 0.4)
    ax[i].set_title(j, loc = 'left', size = 9)

fig.supxlabel('Time Delay')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

tau_all = [6, 8, 4, 6, 4, 4]

fig, ax = plt.subplots(2, 3, sharex = True, figsize = (12, 6))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(cluster_df.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(cluster_df[j], e, tau_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)

fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

E_all = [5, 3, 4, 6, 6, 7]

fig, ax = plt.subplots(2, 3, subplot_kw = dict(projection = '3d'), figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(cluster_df.columns): 
    
    M = mv.build_shadow(cluster_df[j], E_all[i], tau_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, loc = 'left', size = 10)
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)

plt.tight_layout()
plt.show()

# 1) Temperature

L_ranges = [np.arange(38, 90, 5), np.arange(33, 94, 5),
            np.arange(23, 90, 5), np.arange(43, 94, 5),
            np.arange(28, 90, 5), np.arange(33, 90, 5)]
titles = cluster_df.columns

figs = []
for i in range(6):
    fig_ax = ccm_result_1(CCM.CCM, cluster_df.iloc[:, i].values, Temperature.values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'Temperature')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(20, 8))
ax = ax.flatten()

for i in range(6):

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

figs = []
for i in range(6):
    fig_ax = ccm_result_1(CCM.CCM, cluster_df.iloc[:, i].values, RH.values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'RH')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(20, 8))
ax = ax.flatten()

for i in range(6):

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

figs = []
for i in range(6):
    fig_ax = ccm_result_1(CCM.CCM, cluster_df.iloc[:, i].values, Windspeed[1:-1].values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'Windspeed')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(20, 8))
ax = ax.flatten()

for i in range(6):

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

figs = []
for i in range(6):
    fig_ax = ccm_result_1(CCM.CCM, cluster_df.iloc[:, i].values, Precipitation[1:-1].values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'Precipitation')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(20, 8))
ax = ax.flatten()

for i in range(6):

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

# %% CCM for animals and antibiotics

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

def has_consecutive_nans(series, threshold): # Thanks to GenAI
   
    nan_vals = series.isna()
    group = (~nan_vals).cumsum()
    streaks = nan_vals.groupby(group).cumsum()
   
    return (streaks >= threshold).any()

cols_to_drop = [col for col in filtered_animal_ab_table.columns if has_consecutive_nans(filtered_animal_ab_table[col], 5)]
animal_ab_df_cleaned = filtered_animal_ab_table.drop(columns = cols_to_drop)
loss = ((filtered_animal_ab_table.shape[1] - animal_ab_df_cleaned.shape[1]) / filtered_animal_ab_table.shape[1]) * 100
print(f'Loss: {np.round(loss, 2)}%')

#################################################################

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

cols_to_drop = [col for col in filtered_animal_ab_table.columns if has_consecutive_nans(filtered_animal_ab_table[col], 5)]
animal_ab_df_cleaned = filtered_animal_ab_table.drop(columns = cols_to_drop)
loss = ((filtered_animal_ab_table.shape[1] - animal_ab_df_cleaned.shape[1]) / filtered_animal_ab_table.shape[1]) * 100
print(f'Loss: {np.round(loss, 2)}%')

# Run untill this

cols_to_remove = animal_ab_table.columns[(animal_ab_table == 0).sum() >= 60]
animal_ab_df = animal_ab_table.drop(columns = cols_to_remove)

animal_ab_df_processed = preprocessing(animal_ab_df.interpolate(limit_direction = 'both').rolling(window = 5, center = True).mean().dropna())
print(animal_ab_df_processed)

k_vals = np.arange(2, 22, 1)
sse = []
shilloutte = []
centers = []
labels = []

for i in k_vals:
    kmeans = KMeans(n_clusters = i, random_state = 42, n_init = 20)
    kmeans.fit(animal_ab_df_processed.T)
    labels.append(kmeans.labels_)
    centers.append(kmeans.cluster_centers_)
    sse.append(kmeans.inertia_)
    shilloutte.append(silhouette_score(animal_ab_df_processed.T, kmeans.labels_))

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

c = KMeans(n_clusters = 6, random_state = 42, n_init = 20).fit(animal_ab_df_processed.T)
c_l = c.labels_
        
coordinates_plot(animal_ab_df_processed.T.values, c_l, )

c_l = np.array(c_l)
cols = animal_ab_df_processed.columns
colors = cm.tab10.colors

fig, ax = plt.subplots(6, 1, figsize = (10, 12), sharex = True)
ax = ax.flatten()
cluster_means = {}
for i, clust in enumerate(np.unique(c_l)):
    
    selected_cols = cols[c_l == clust]
    data = animal_ab_df_processed[selected_cols.values]
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

tsne = TSNE(n_components = 2, perplexity = 4, random_state=42)
X_new = tsne.fit_transform(animal_ab_df_processed.T)
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

animal_cluster_df = pd.DataFrame(cluster_means)

# CCM for 6 clusters

fig, ax = plt.subplots(2, 3, sharex = True, figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(animal_cluster_df.columns):
    mutual_info = tdmi.tdmi(animal_cluster_df[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_title(j, loc = 'left', size = 9)
    
fig.supxlabel('Time Delay')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

tau_6_all = [5, 4, 5, 4, 5, 5]

fig, ax = plt.subplots(2, 3, sharex = True, figsize = (12, 6))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(animal_cluster_df.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(animal_cluster_df[j], e, tau_6_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)

fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

E_6_all = [5, 5, 6, 4, 6, 5]

fig, ax = plt.subplots(2, 3, subplot_kw = dict(projection = '3d'), figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(animal_cluster_df.columns): 
    
    M = mv.build_shadow(animal_cluster_df[j], E_6_all[i], tau_6_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, loc = 'left', size = 10)
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)

plt.tight_layout()
plt.show()

L_6_ranges = [np.arange(33, 90, 5), np.arange(28, 90, 5),
              np.arange(33, 90, 5), np.arange(28, 90, 5),
              np.arange(38, 90, 5), np.arange(33, 90, 5),]

# 1. Temperature

figs = []
for i in range(6):
    fig_ax = ccm_result_1(CCM.CCM, animal_cluster_df.iloc[:, i].values, Temperature.values, L_6_ranges[i], 
                          4, tau_6_all[i], 4, E_6_all[i])
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(20, 8))
ax = ax.flatten()

for i in range(6):

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

# 2. Relative Humidity

figs = []
for i in range(6):
    fig_ax = ccm_result_1(CCM.CCM, animal_cluster_df.iloc[:, i].values, RH.values, L_6_ranges[i], 
                          4, tau_6_all[i], 4, E_6_all[i])
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(20, 8))
ax = ax.flatten()

for i in range(6):

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

# 3. Windspeed
figs = []
for i in range(6):
    fig_ax = ccm_result_1(CCM.CCM, animal_cluster_df.iloc[:, i].values, Windspeed[1:-1].values, L_6_ranges[i], 
                          3, tau_6_all[i], 4, E_6_all[i])
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(20, 8))
ax = ax.flatten()

for i in range(6):

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

# 4. Precipitation

figs = []
for i in range(6):
    fig_ax = ccm_result_1(CCM.CCM, animal_cluster_df.iloc[:, i].values, Precipitation[1:-1].values, L_6_ranges[i], 
                          3, tau_6_all[i], 4, E_6_all[i])
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(20, 8))
ax = ax.flatten()

for i in range(6):

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

# Comparing 6, 7, and 8

labeled_df = pd.DataFrame(dict(name = animal_ab_df_processed.columns, labels = c_l))
cluster_61 = labeled_df[labeled_df['labels'] == 1]
cluster_63 = labeled_df[labeled_df['labels'] == 3]

cluster_71 = cluster_61.copy()
cluster_73_1 = [('Calves', 'Azithromycine'), ('Calves', 'Gentamicin')]
cluster_73_2 = [('Calves', 'Chloramphenicol'), ('Calves', 'Ciprofloxacin'), ('Calves', 'Nalidixic Acid')]

cluster_81_1 = [('Poultry', 'Azithromycine')]
cluster_81_2 = [('Pigs', 'Ciprofloxacin'), ('Pigs', 'Nalidixic Acid'), ('Poultry', 'Gentamicin')]
cluster_83_1 = [('Calves', 'Azithromycine'), ('Calves', 'Gentamicin')]
cluster_83_2 = [('Calves', 'Chloramphenicol'), ('Calves', 'Ciprofloxacin'), ('Calves', 'Nalidixic Acid')]

cluster_8_df = pd.DataFrame({
    'cluster_81_1': animal_ab_df_processed[cluster_81_1].mean(axis = 1),
    'cluster_81_2': animal_ab_df_processed[cluster_81_2].mean(axis = 1),
    'cluster_83_1': animal_ab_df_processed[cluster_83_1].mean(axis = 1),
    'cluster_83_2': animal_ab_df_processed[cluster_83_2].mean(axis = 1)})

fig, ax = plt.subplots(2, 2, sharex = True, figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(cluster_8_df.columns):
    mutual_info = tdmi.tdmi(cluster_8_df[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', lw = 1.2)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_title(j, loc = 'left', size = 9)
    
fig.supxlabel('Time Delay')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

tau_4_all = [4, 5, 4, 5]

fig, ax = plt.subplots(2, 2, sharex = True, figsize = (12, 6))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(cluster_8_df.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(cluster_8_df[j], e, tau_4_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4)
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)

fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

E_4_all = [5, 5, 4, 6]

fig, ax = plt.subplots(2, 2, subplot_kw = dict(projection = '3d'), figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(cluster_8_df.columns): 
    
    M = mv.build_shadow(cluster_8_df[j], E_4_all[i], tau_4_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, loc = 'left', size = 10)
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)

plt.tight_layout()
plt.show()

L_4_ranges = [np.arange(28, 90, 5), np.arange(33, 90, 5),
              np.arange(23, 90, 5), np.arange(38, 90, 5),]

# 1. Temperature

figs = []
for i in range(4):
    fig_ax = ccm_result_1(CCM.CCM, cluster_8_df.iloc[:, i].values, Temperature.values, L_4_ranges[i], 
                          4, tau_4_all[i], 4, E_4_all[i])
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(20, 8))
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

plt.tight_layout()
plt.show()

# 2. Relative Humidity

figs = []
for i in range(4):
    fig_ax = ccm_result_1(CCM.CCM, cluster_8_df.iloc[:, i].values, RH.values, L_4_ranges[i], 
                          4, tau_4_all[i], 4, E_4_all[i])
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(20, 8))
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

plt.tight_layout()
plt.show()

# 3. Windspeed
figs = []
for i in range(4):
    fig_ax = ccm_result_1(CCM.CCM, cluster_8_df.iloc[:, i].values, Windspeed[1:-1].values, L_4_ranges[i], 
                          4, tau_4_all[i], 4, E_4_all[i])
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(20, 8))
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

plt.tight_layout()
plt.show()

# 4. Precipitation

figs = []
for i in range(4):
    fig_ax = ccm_result_1(CCM.CCM, cluster_8_df.iloc[:, i].values, Precipitation[1:-1].values, L_4_ranges[i], 
                          3, tau_4_all[i], 4, E_4_all[i])
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(20, 8))
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

plt.tight_layout()
plt.show()

# %% Testing out Calves Tetracycline

calves_tetracycline = animal_ab_df_processed[( 'Calves',     'Tetracycline')]
fig, ax = plt.subplots(figsize = (8, 6))

ax.plot(calves_tetracycline.index, calves_tetracycline.values, 'o-',)
ax.grid()
ax.set_xlabel('Time [Months]')
ax.set_ylabel('Resistance (Normalised)')
ax.set_title('Tetracycline Resistant E.coli occurance in Calves')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 6))

mutual_info = tdmi.tdmi(calves_tetracycline, 9, 4)
ax.plot(np.arange(1, 10, 1), mutual_info, 'o-')
ax.grid()
ax.set_xlabel('Time Delay')
ax.set_ylabel('Mutual Information')
ax.set_title('Averge Mutual Information')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 6))
opt_E = []
for e in np.arange(1, max_E):

    r = afn.afn(calves_tetracycline, e, 6, 'euclidean', 1, None)
    opt_E.append(np.asarray(r).T)
    
E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
ax.plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4)
ax.grid('--', alpha = 0.4, color = 'grey')
ax.set_title('Choosing Optimal Embedding Dimension', fontsize=11)
ax.set_xlabel('Embedding Dimesnion')
ax.set_ylabel('E1 Score')
plt.tight_layout()
plt.show()

M = mv.build_shadow(calves_tetracycline, 3, 6)
fig, ax = plt.subplots(figsize = (6, 4), subplot_kw = {'projection': '3d'})

ax.plot(M[:,0], M[:, 1], M[:, 2], '.-', lw = 1.2)
ax.set_xlabel('Resistance (Normalised) (t)')
ax.set_ylabel('Resistance (Normalised) (t-tau)')
ax.set_zlabel('Resistance (Normalised) (t-2tau)')
ax.set_title('Reconstructed Shadow Manifold \nof Tetracycline Resistance in Calves')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 2, sharex = True, )
ax = ax.flatten()
L_range = np.arange(23, 90, 5)
temp_climate_df = pd.DataFrame(dict(Temperature = Temperature.values, 
                                 Windspeed = Windspeed[1:-1].values, 
                                 RH = RH.values,
                                 Precipitation = Precipitation[1:-1].values))

figs = []
for i in range(4):
    fig_ax = ccm_result_1(CCM.CCM, calves_tetracycline.values, temp_climate_df.iloc[:, i].values, L_range, 
                          4, 6, 4, 3, temp_climate_df.columns[i])
    fig_ax.set_title(temp_climate_df.columns[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
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
fig.suptitle('Tetracycline Resistant E. coli occurance in Calves')
plt.tight_layout()
plt.show()

# %% Extended CCM

# Try out in calves-tetracycline

figs = []
for i in range(4):
    fig_ax = extended_ccm.extended_ccm(CCM.CCM, calves_tetracycline.values, temp_climate_df.iloc[:, i].values, 90, 
                          4, 6, 4, 3, 3,temp_climate_df.columns[i])
    fig_ax.set_title(temp_climate_df.columns[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
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
fig.suptitle('Tetracycline Resistant E. coli occurance in Calves')
plt.tight_layout()
plt.show() 

# Extended CCM for All ABs + MDR
# 1. Temperature

tau_all = [4, 4, 4, 4, 4, 5, 6, 5, 6, 6, 5, 6, 8, 6]
E_all = [5, 4, 4, 7, 6, 5, 6, 4, 5, 4, 5, 5, 3, 4]

figs = []
for i in range(14):
    fig_ax = extended_ccm.extended_ccm(CCM.CCM, amr_df_processed_one.iloc[:,i].values, Temperature.values, 90, 
                          4, tau_all[i], 4, E_all[i], tp = 3, variable = 'Temperature')
    fig_ax.set_title(amr_df_processed_one.columns[i])
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

# 2. Relative Humidity

figs = []
for i in range(14):
    fig_ax = extended_ccm.extended_ccm(CCM.CCM, amr_df_processed_one.iloc[:,i].values, RH.values, 90, 
                          4, tau_all[i], 4, E_all[i], tp = 3, variable = 'RH')
    fig_ax.set_title(amr_df_processed_one.columns[i])
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

figs = []
for i in range(14):
    fig_ax = extended_ccm.extended_ccm(CCM.CCM, amr_df_processed_one.iloc[:,i].values, Windspeed[1:-1].values, 90, 
                          4, tau_all[i], 4, E_all[i], tp = 3, variable = 'Windspeed')
    fig_ax.set_title(amr_df_processed_one.columns[i])
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

figs = []
for i in range(14):
    fig_ax = extended_ccm.extended_ccm(CCM.CCM, amr_df_processed_one.iloc[:,i].values, Precipitation[1:-1].values, 90, 
                          3, tau_all[i], 4, E_all[i], tp = 3, variable = 'Precipitation')
    fig_ax.set_title(amr_df_processed_one.columns[i])
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
# %% Animal AB new

final_animal_table = preprocessing(animal_ab_table_filtered.interpolate().rolling(window = 5, center = True).mean().dropna())
k_vals = np.arange(2, 15, 1)
sse = []
shilloutte = []
centers = []
labels = []

for i in k_vals:
    kmeans = KMeans(n_clusters = i, random_state = 42, n_init = 20)
    kmeans.fit(final_animal_table.T)
    labels.append(kmeans.labels_)
    centers.append(kmeans.cluster_centers_)
    sse.append(kmeans.inertia_)
    shilloutte.append(silhouette_score(final_animal_table.T, kmeans.labels_))

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
plt.show()

c = KMeans(n_clusters = 5, random_state = 42, n_init = 20).fit(final_animal_table.T)
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
        
coordinates_plot(final_animal_table.T.values, c_l, )

c_l = np.array(c_l)
cols = final_animal_table.columns
colors = cm.tab10.colors

fig, ax = plt.subplots(5, 1, figsize = (10, 12), sharex = True)
ax = ax.flatten()
cluster_means = {}
for i, clust in enumerate(np.unique(c_l)):
    
    selected_cols = cols[c_l == clust]
    data = final_animal_table[selected_cols.values]
    cluster_means[f'clust_{i +1}'] = np.mean(data, axis=1)

    for col in data.columns:
        ax[i].plot(data.index, data[col], alpha = 0.15, color = colors[i],)

    ax[i].plot(data.index, np.mean(data, axis = 1), alpha = 0.9, color = colors[i], label = f'Cluster {i + 1} mean')
    ax[i].set_xmargin(0)
    ax[i].set_xticks(data.index[::6])
    ax[i].tick_params(axis = 'x', rotation = 90)
    ax[i].legend(loc = 'upper right')
    ax[i].set_title(f'Cluster {i + 1}', loc = 'left', size = 8.5)
    ax[i].grid()

fig.supylabel('Resistance (%) (Normalised)')
fig.supxlabel('Time [Month]')
fig.suptitle('Antimicrobial Resistance in Food Producing Animals')
plt.tight_layout()
plt.show()

tsne = TSNE(n_components = 2, perplexity = 6, random_state=42)
X_new = tsne.fit_transform(final_animal_table.T)
print(f'KL divergence: {tsne.kl_divergence_:.4f}')

X_tsne = pd.DataFrame(X_new, columns=['TSNE1', 'TSNE2'], index=cols)
X_tsne['Cluster'] = c_l

fig, ax = plt.subplots(figsize=(10, 8))

for cluster in sorted(X_tsne['Cluster'].unique()):

    data = X_tsne[X_tsne['Cluster'] == cluster]
    ax.scatter(data['TSNE1'], data['TSNE2'], s=25,
        color=colors[int(cluster) % len(colors)], label=f'Cluster {cluster +1}')
    for idx, row in data.iterrows():
        ax.text(row['TSNE1'] + 0.1, row['TSNE2'] + 0.1, idx, fontsize=8.5)

ax.set_title('t-SNE projection of AMR Occurance in Food Producing \nAnimals in Belgium coloured by K-means clusters', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.grid(alpha=0.3)
ax.legend(title='Cluster')
plt.show()

cluster_df = pd.DataFrame(cluster_means)
print(cluster_df)

fig, ax = plt.subplots(2, 3, figsize = (12, 8), sharex = True)
ax = ax.flatten()

for i, j in enumerate(cluster_df.columns):
    ax[i].plot(cluster_df.index, cluster_df[j], '.-', color = 'blue',)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_title(j, fontsize = 9)
    
ax[len(cluster_df.columns) - 3].tick_params('x', labelbottom = True)
fig.delaxes(ax[len(cluster_df.columns)])
fig.supxlabel('Time [Months]')
fig.supylabel('Resistance (%) (Normalised)')
fig.suptitle('AntiMicrobial Resistance Patterns across different clusters')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 3, figsize = (12, 8), sharex = True)
ax = ax.flatten()

for i, j in enumerate(cluster_df.columns):
    mutual_info = tdmi.tdmi(cluster_df[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', color = 'blue',)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_title(j, fontsize = 9)
    
ax[len(cluster_df.columns) - 3].tick_params('x', labelbottom = True)
fig.delaxes(ax[len(cluster_df.columns)])
fig.supxlabel('Time Delay')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

tau_all = [8, 5, 5, 5, 6]

fig, ax = plt.subplots(2, 3, sharex = True, figsize = (12, 8))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(cluster_df.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(cluster_df[j], e, tau_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4, c = 'blue')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)
ax[len(cluster_df.columns) - 3].tick_params('x', labelbottom = True)
fig.delaxes(ax[len(cluster_df.columns)])
fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

E_all = [5, 5, 6, 5, 5]

fig, ax = plt.subplots(2, 3, subplot_kw = dict(projection = '3d'), figsize = (12, 8))
ax = ax.flatten()

for i, j in enumerate(cluster_df.columns): 
    
    M = mv.build_shadow(cluster_df[j], E_all[i], tau_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], c = 'blue', lw = 1.2)
    ax[i].set_title(j, loc = 'left', size = 10)
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)
ax[len(cluster_df.columns) - 3].tick_params('x', labelbottom = True)
fig.delaxes(ax[len(cluster_df.columns)])
plt.tight_layout()
plt.show()

L_ranges = [np.arange(48, 90, 5), np.arange(33, 90, 5),
              np.arange(38, 90, 5), np.arange(33, 90, 5),
              np.arange(38, 90, 5)]
titles = cluster_df.columns
# 1. Temperature

figs = []
for i in range(5):
    fig_ax = ccm_result_1(CCM.CCM, cluster_df.iloc[:, i].values, Temperature.values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'Temperature')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax = ax.flatten()

for i in range(5):

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
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
        
ax[len(cluster_df.columns) - 3].tick_params('x', labelbottom = True)
fig.delaxes(ax[len(cluster_df.columns)])
fig.supxlabel('Time Series Length [L]')
fig.supylabel('CCM Skill (\rho)')
fig.suptitle('Convergent Cross Mapping of AMR and Temperature')
plt.tight_layout()
plt.show()

# 2. Relative Humidity

figs = []
for i in range(5):
    fig_ax = ccm_result_1(CCM.CCM, cluster_df.iloc[:, i].values, RH.values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'RH')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax = ax.flatten()

for i in range(5):

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
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
ax[len(cluster_df.columns) - 3].tick_params('x', labelbottom = True)
fig.delaxes(ax[len(cluster_df.columns)])
fig.supxlabel('Time Series Length [L]')
fig.supylabel('CCM Skill (\rho)')
fig.suptitle('Convergent Cross Mapping of AMR and Relative Humidity')
plt.tight_layout()
plt.show()

# 3. Windspeed
figs = []
for i in range(5):
    fig_ax = ccm_result_1(CCM.CCM, cluster_df.iloc[:, i].values, Windspeed[1:-1].values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'Windspeed')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax = ax.flatten()

for i in range(5):

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
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
ax[len(cluster_df.columns) - 3].tick_params('x', labelbottom = True)
fig.delaxes(ax[len(cluster_df.columns)])
fig.supxlabel('Time Series Length [L]')
fig.supylabel('CCM Skill (\rho)')
fig.suptitle('Convergent Cross Mapping of AMR and Windspeed')
plt.tight_layout()
plt.show()

# 4. Precipitation

figs = []
for i in range(5):
    fig_ax = ccm_result_1(CCM.CCM, cluster_df.iloc[:, i].values, Precipitation[1:-1].values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'Precipitation')
    fig_ax.set_title(titles[i])
    fig_ax.grid('--', c='grey', alpha=0.4)

    figs.append(fig_ax)

plt.show()

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax = ax.flatten()

for i in range(5):

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
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
ax[len(cluster_df.columns) - 3].tick_params('x', labelbottom = True)
fig.delaxes(ax[len(cluster_df.columns)])
fig.supxlabel('Time Series Length [L]')
fig.supylabel('CCM Skill (\rho)')
fig.suptitle('Convergent Cross Mapping of AMR and Precipitation')
plt.tight_layout()
plt.show()

# %% New Table Alone

fig, ax = plt.subplots(3, 6, sharex = True, figsize = (24, 12))
ax = ax.flatten()

for i, j in enumerate(final_animal_table.columns):
    ax[i].plot(final_animal_table.index, final_animal_table[j], '.-', c = 'blue')
    ax[i].set_title(j, fontsize = 11)
    ax[i].grid('--', c = 'grey', alpha = 0.4)

fig.supxlabel('Time [Months]')
fig.supylabel('Resistance (%) (Normalised)')
fig.suptitle('E coli resistance occurance in animals in Belgium from 2017-2024')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(3, 6, figsize = (24, 12), sharex = True)
ax = ax.flatten()

for i, j in enumerate(final_animal_table.columns):
    mutual_info = tdmi.tdmi(final_animal_table[j], 9, 4)
    ax[i].plot(np.arange(1, 10, 1), mutual_info, 'o-', color = 'blue',)
    ax[i].grid('--', c = 'grey', alpha = 0.3)
    ax[i].set_title(j, fontsize = 9)
    
fig.supxlabel('Time Delay')
fig.supylabel('Mutual Information')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

tau_all = [8, 5, 5, 8, 5, 7, 
           8, 5, 5, 5, 5, 6,
           5, 5, 5, 5, 5, 6]

fig, ax = plt.subplots(3, 6, sharex = True, figsize = (24, 12,))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(final_animal_table.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(final_animal_table[j], e, tau_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-', lw = 1.2, markersize = 4, c = 'blue')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)

fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

E_all = [4, 7, 6, 5, 7, 5, 
         5, 4, 5, 5, 5, 5, 
         5, 4, 4, 6, 5, 5]

fig, ax = plt.subplots(3, 6, subplot_kw = dict(projection = '3d'), figsize = (24, 12))
ax = ax.flatten()

for i, j in enumerate(final_animal_table.columns): 
    
    M = mv.build_shadow(final_animal_table[j], E_all[i], tau_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], c = 'blue', lw = 1.2)
    ax[i].set_title(j, loc = 'left', size = 10)
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 7, pad = 0)
    ax[i].tick_params(axis = 'z', which = 'major', labelsize = 7, pad = 0)
plt.tight_layout()
plt.show()

L_ranges = [np.arange(38, 90, 5), np.arange(43, 90, 5),
    np.arange(38, 90, 5), np.arange(48, 90, 5),
    np.arange(43, 90, 5), np.arange(43, 90, 5),
    np.arange(48, 90, 5), np.arange(28, 90, 5),
    np.arange(33, 90, 5), np.arange(33, 90, 5),
    np.arange(33, 90, 5), np.arange(38, 90, 5),
    np.arange(33, 90, 5), np.arange(28, 90, 5),
    np.arange(28, 90, 5), np.arange(38, 90, 5),
    np.arange(33, 90, 5), np.arange(38, 90, 5),]

titles = final_animal_table.columns

# 1. Temperature

figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, final_animal_table.iloc[:, i].values, Temperature.values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'Temperature')
    fig_ax.set_title(titles[i])
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

    ax[i].set_title(source_ax.get_title(loc='center'))
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()
        
fig.supxlabel('Time Series Length [L]')
fig.supylabel('CCM Skill (\rho)')
fig.suptitle('Convergent Cross Mapping of AMR and Temperature')
plt.tight_layout()
plt.show()

# 2. Relative Humidity

figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, final_animal_table.iloc[:, i].values, RH.values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'RH')
    fig_ax.set_title(titles[i])
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

    ax[i].set_title(source_ax.get_title(loc='center'))
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()

fig.supxlabel('Time Series Length [L]')
fig.supylabel('CCM Skill (\rho)')
fig.suptitle('Convergent Cross Mapping of AMR and Relative Humidity')
plt.tight_layout()
plt.show()

# 3. Windspeed
figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, final_animal_table.iloc[:, i].values, Windspeed[1:-1].values, L_ranges[i], 
                          4, tau_all[i], 4, E_all[i], variable = 'Windspeed')
    fig_ax.set_title(titles[i])
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

    ax[i].set_title(source_ax.get_title(loc='center'))
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()

fig.supxlabel('Time Series Length [L]')
fig.supylabel('CCM Skill (\rho)')
fig.suptitle('Convergent Cross Mapping of AMR and Windspeed')
plt.tight_layout()
plt.show()

# 4. Precipitation

figs = []
for i in range(18):
    fig_ax = ccm_result_1(CCM.CCM, final_animal_table.iloc[:, i].values, Precipitation[1:-1].values, L_ranges[i], 
                          3, tau_all[i], 4, E_all[i], variable = 'Precipitation')
    fig_ax.set_title(titles[i])
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

    ax[i].set_title(source_ax.get_title(loc='center'))
    ax[i].grid('--', c='grey', alpha=0.4)

    if len(source_ax.get_legend_handles_labels()[0]) > 0:

        ax[i].legend()

fig.supxlabel('Time Series Length [L]')
fig.supylabel('CCM Skill (\rho)')
fig.suptitle('Convergent Cross Mapping of AMR and Precipitation')
plt.tight_layout()
plt.show()

# MultiIndex([(   'Pigs',       'Ampicillin'),
#             (   'Pigs',    'Azithromycine'),
#             (   'Pigs',       'Cefotaxime'),
#             (   'Pigs',      'Ceftazidime'),
#             (   'Pigs',  'Chloramphenicol'),
#             (   'Pigs',    'Ciprofloxacin'),
#             (   'Pigs',         'Colistin'),
#             (   'Pigs',       'Gentamicin'),
#             (   'Pigs',       'Meropeneme'),
#             (   'Pigs',   'Nalidixic Acid'),
#             (   'Pigs', 'Sulfamethoxazole'),
#             (   'Pigs',     'Tetracycline'),
#             (   'Pigs',      'Tigecycline'),
#             (   'Pigs',     'Trimethoprim'),
#             ('Poultry',       'Ampicillin'),
#             ('Poultry',    'Azithromycine'),
#             ('Poultry',       'Cefotaxime'),
#             ('Poultry',      'Ceftazidime'),
#             ('Poultry',  'Chloramphenicol'),
#             ('Poultry',    'Ciprofloxacin'),
#             ('Poultry',         'Colistin'),
#             ('Poultry',       'Gentamicin'),
#             ('Poultry',       'Meropeneme'),
#             ('Poultry',   'Nalidixic Acid'),
#             ('Poultry', 'Sulfamethoxazole'),
#             ('Poultry',     'Tetracycline'),
#             ('Poultry',      'Tigecycline'),
#             ('Poultry',     'Trimethoprim')],
#            names=['Animal Type', 'Active Substance'])

# %% Iso res and MDR and 18CCM

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

# Things to test a) 18 ABs b) 9 MDRs

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

# CCM Parameters

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

# %% 
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
