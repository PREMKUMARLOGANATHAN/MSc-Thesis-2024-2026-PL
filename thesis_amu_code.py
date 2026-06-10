# %% Author Note

'''

This code is written by Prem Kumar Loganathan for the MSc Thesis titled, 
'Impact of Climate Change on AntiMicrobial Resistance'

All the class objects and user-defined functions needed for this code is made available in the same repository. 
No explanation will be provided anywhere in code until and unless necessary.

“When I wrote this code, only God and I understood what I did. Now only God knows. Best of luck figuring out!”

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

animal_name_map = {'PIG': 'Pigs',
                   'VECLF': 'Calves',
                   'PLTR': 'Poultry'}

amu_df['Active_Substance'] = amu_df['Active_Substance'].replace(ab_name)
amu_df['AnimalType'] = amu_df['AnimalType'].replace(animal_name_map)

amu_df_for_clustering = amu_df.groupby(['YY-MM', 'Active_Substance', 'AnimalType'])[['Total_Active_Substance']].sum().reset_index()
table_for_clustering = pd.pivot_table(data = amu_df_for_clustering, index = 'YY-MM', values = 'Total_Active_Substance', columns = ['AnimalType', 'Active_Substance'])

def has_consecutive_nans(series, threshold): # thanks to Gen AI
    
    nan_vals = series.isna()
    group = (~nan_vals).cumsum()
    streaks = nan_vals.groupby(group).cumsum()
    
    return (streaks >= threshold).any()

cols_to_drop = [col for col in table_for_clustering.columns if has_consecutive_nans(table_for_clustering[col], 5)]
amu_df_cleaned = table_for_clustering.drop(columns = cols_to_drop)
more_than_6_months_missing = amu_df_cleaned.isna().sum()>6
cols_missing = more_than_6_months_missing[more_than_6_months_missing].index.tolist()
cols_to_drop = list(set(cols_to_drop + cols_missing))
amu_df_cleaned = table_for_clustering.drop(columns = cols_to_drop)
loss = ((table_for_clustering.shape[1] - amu_df_cleaned.shape[1]) / table_for_clustering.shape[1]) * 100
print(f'Loss: {np.round(loss, 2)}%')

# Gaps checking

fig, ax = plt.subplots(2, 1, figsize = (7, 4))
ax1, ax2 = ax

ax1.plot(table_for_clustering.index, table_for_clustering[cols_to_drop], '.', color = 'grey', alpha = 0.6)
ax2.plot(table_for_clustering.index, table_for_clustering[cols_missing], '.', color = 'red', alpha = 0.7)

amu_df_filled = preprocessing(amu_df_cleaned.interpolate())

k_vals = np.arange(2, 19, 1)
sse = []
shilloutte = []
centers = []
labels = []

for i in k_vals:
    kmeans = KMeans(n_clusters = i, random_state = 42, n_init = 20)
    kmeans.fit(amu_df_filled.T)
    labels.append(kmeans.labels_)
    centers.append(kmeans.cluster_centers_)
    sse.append(kmeans.inertia_)
    shilloutte.append(silhouette_score(amu_df_filled.T, kmeans.labels_))

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

c = KMeans(n_clusters = 8, random_state = 42, n_init = 20).fit(amu_df_filled.T)
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
colors = cm.Set1.colors

fig, ax = plt.subplots(8, 1, figsize = (10, 12), sharex = True)
ax = ax.flatten()
cluster_means = {}
for i, clust in enumerate(np.unique(c_l)):
    
    selected_cols = cols[c_l == clust]
    data = amu_df_filled[selected_cols.values]
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

fig.supylabel('Total Active Substance (Normalised)', x = 0.04)
fig.supxlabel('Time [Month]', y = -0.05)
fig.suptitle('Antimicrobial Usage in Food Producing Animals', y = 0.97)

tsne = TSNE(
    n_components=2,
    perplexity=17,
    random_state=42)

X_new = tsne.fit_transform(amu_df_filled.T)

print(f'KL divergence: {tsne.kl_divergence_}')

X_tsne = pd.DataFrame(X_new, columns=['TSNE1', 'TSNE2'], index=cols)
X_tsne['Cluster'] = c_l

fig, ax = plt.subplots(figsize=(10, 8))

for cluster in sorted(X_tsne['Cluster'].unique()):

    data = X_tsne[X_tsne['Cluster'] == cluster]
    ax.scatter(data['TSNE1'], data['TSNE2'], s=25,
        color=colors[int(cluster) % len(colors)], label=f'Cluster {cluster}')
    for idx, row in data.iterrows():
        ax.text(row['TSNE1'] + 0.1, row['TSNE2'] + 0.1, cluster + 1, fontsize=8.5)

ax.set_title(
    't-SNE projection of AMU in Food Producing \nAnimals in Belgium coloured by K-means clusters',
    fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.grid(alpha=0.3)
ax.legend(title='Cluster')

# Colored by Animal

animals = X_tsne.index.get_level_values(0)
unique_animals = animals.unique()

animal_colors = {animal: colors[i % len(colors)] for i, animal in enumerate(unique_animals)}

fig, ax = plt.subplots(figsize=(10, 8))

for animal in unique_animals:

    data = X_tsne[animals == animal]

    ax.scatter( data['TSNE1'], data['TSNE2'], s=25,
        color=animal_colors[animal], label=animal)

    for idx, row in data.iterrows():
        ax.text(row['TSNE1'] + 0.1, row['TSNE2'] + 0.1, str(int(row['Cluster'] + 1)), fontsize=8.5)
ax.set_title( 't-SNE projection of AMU in Food Producing \nAnimals in Belgium coloured by Animal Types',
             fontsize = 12, weight = 'bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.grid(alpha=0.3)
ax.legend(title='Animal')

# %% AMU E coli

ABs = [( 'Calves',               'Ampicillin'), (   'Pigs',               'Ampicillin'), (   'Pigs',               'Gentamicin')]
fig, ax = plt.subplots(1, 1,)
ax.plot(amu_df_filled.index, amu_df_filled[ABs], label = ABs,)
ax.set_xticks(amu_df_filled.index[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', alpha = 0.4)
ax.legend()

# %% Why clustering bring them together

Nearby1 = [('Pigs', 'Apramycin'), ('Calves', 'Tulathromycin'), ('Poultry', 'Colistin'), ('Pigs', 'Neomycin'), ('Calves', 'Trimethoprim_Sulfonamide')]
labels = [f'{animal} - {drug}' for animal, drug in Nearby1]

fig, ax = plt.subplots(figsize=(12, 6))

for col, label in zip(Nearby1, labels):
    ax.plot(amu_df_filled.index,
        amu_df_filled[col], alpha = 0.9,
        linewidth=2, label=label)

ax.set_title('Antimicrobial Usage in Food Producing Animals in Belgium')

ax.set_xlabel('Time [Months]')
ax.set_ylabel('Normalised Active Substance Consumed')

ax.grid(True, alpha=0.25)
ax.set_xticks(amu_df_filled.index[4::6])
ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    frameon=False,
    title='Animal - Antimicrobial')
ax.tick_params('x', rotation = 90)
ax.set_xmargin(0)
plt.tight_layout()
plt.show()

# %% Investigating Causal Relationship using Correlation Coefficient

single_df = pd.DataFrame({
    'Full AMU': amu_df_filled.mean(axis=1),
    'AMU Clust 1': cluster_means['clust_0'],
    'AMU Clust 2': cluster_means['clust_1'],
    'AMU Clust 3': cluster_means['clust_2'],
    'Temperature': preprocessing(temp.values),
    'Relative Humidity': preprocessing(rh.values),
    'Precipitation': preprocessing(prec.values),
    'Wind Speed': preprocessing(ws.values)}, index = amu_df_filled.index)

corr = single_df.corr(method="pearson")
mask = np.tril(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, mask = mask, cmap = "RdBu_r", vmin = -1, vmax = 1,
    center = 0, annot = True, fmt = ".2f", annot_kws = {"size": 9},
    square = True, linewidths = 0.5,linecolor = "white",
    cbar_kws = dict(shrink = 0.3, label = 'Correalation Coefficient',
                   orientation = 'horizontal', location = 'bottom', pad = 0.01), ax = ax)

rect = patches.Rectangle(xy = (4, 0), width = 4, height = 4,
                         fill = False, edgecolor = 'k', lw = 2)
ax.add_patch(rect)
ax.text(1, 2, 'AMU', color = 'k', weight = 'bold', bbox = dict(facecolor='none', edgecolor = 'k'))
ax.text(2.4, 4.3, 'AMU & \nClimate', color = 'k', weight = 'bold', bbox = dict(facecolor='none', edgecolor = 'k'))
ax.text(4.5, 6.5, 'Climate', color = 'k', weight = 'bold', bbox = dict(facecolor='none', edgecolor = 'k'))

ax.tick_params(axis="x", rotation=90, bottom=False,
    top=True, labelbottom=False, labeltop=True)
ax.tick_params(axis="y", rotation=0, left = False, 
    right = True, labelleft = False, labelright = True)
plt.setp(ax.get_xticklabels(), ha="center")
plt.setp(ax.get_yticklabels(), va ='center')
plt.tight_layout()
plt.show()

# %% Investigating Causal Relationship using CCM

# Noisy
# 2 x 4 ACF

fig, ax = plt.subplots(2, 4, sharex = True, sharey = True, figsize = (10, 6))
ax = ax.flatten()

for i, j in enumerate(single_df.columns):
    plot_acf(single_df[j], lags = 25, ax = ax[i], title = j, 
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

fig, ax = plt.subplots(2, 4, sharex = True, figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(single_df.columns):
    
    mutual_info = tdmi.tdmi(single_df[j], 9, 4)
    ax[i].plot(np.arange(1,10,1), mutual_info, 'o-', )
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)
    
fig.supylabel('Mutual Information')
fig.supxlabel('Time Lags')
fig.suptitle('Average Mutual Information')
plt.tight_layout()
plt.show()

tau_all = [6, 2, 2, 2, 4, 3, 2, 4]

# 2 x 4 Cao's FNN

fig, ax = plt.subplots(2, 4, sharex = True, figsize = (12, 6))
ax = ax.flatten()
max_E = 11

for i, j in enumerate(single_df.columns):
    
    opt_E = []
    for e in np.arange(1, max_E):

        r = afn.afn(single_df[j], e, tau_all[i], 'euclidean', 1, None)
        opt_E.append(np.asarray(r).T)
        
    E1 = [opt_E[i][0] / opt_E[i-1][0] for i in range(1, len(opt_E))]
    ax[i].plot(np.arange(1, max_E-1), E1, 'o-')
    ax[i].axhline(y = 0.9, ls = '--', label = 'threshold', color = 'grey')
    ax[i].grid('--', alpha = 0.4, color = 'grey')
    ax[i].set_title(j, fontsize=11)
    
fig.supylabel('E1 Score')
fig.supxlabel('No. of Embedding Dimension')
fig.suptitle("Cao's FNN for choosing optimal embedding dimension")
plt.tight_layout()
plt.show()

E_all = [6, 5, 5, 5, 4, 5, 5, 4]

# 2 x 4 shadow manifold

fig, ax = plt.subplots(2, 4, subplot_kw = dict(projection = '3d'), figsize = (12, 6))
ax = ax.flatten()

for i, j in enumerate(single_df.columns): 
    
    M = mv.build_shadow(single_df[j], E_all[i], tau_all[i])
    ax[i].plot(M[:,0], M[:,1], M[:,2], lw = 1.2)
    ax[i].set_title(j, size = 9)

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
        
    return line_plots

ani = FuncAnimation(
    fig, 
    update, 
    frames=total_frames, 
    interval=40, 
    blit=False)

plt.tight_layout()

ani.save('evolving_manifold_rotation.gif', writer='pillow', fps=25)
plt.close(fig)'''
