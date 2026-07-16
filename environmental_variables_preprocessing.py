# %% Author Note

'''

This code is written by Prem Kumar Loganathan for the MSc Thesis titled, 
'Environmental Drivers of AntiMicrobial Resistance'

All the class objects and user-defined functions needed for this code is made available in the same repository. 
No explanation will be provided anywhere in code until and unless necessary.

Note: Given the size of the data files, this script processes every variable, individually.

'''

# %% Importing Libraries

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rio
import netCDF4
import h5netcdf
import warnings 

warnings.filterwarnings('ignore')
url = 'https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip'

# %% 2m Air Temperature Preprocessing

temp_2m_path = r'C:\Users\r1015311\Desktop\Climate Data\Agrometerological Indicators\2m Temperature\*.nc'
temp_2m = xr.open_mfdataset(temp_2m_path, engine = 'netcdf4') # Loading Dataset

temp_2m_be = temp_2m.sel(lat = slice(52.5, 47.5),lon = slice(1.75, 6.9), time = slice('2017-01-01', '2024-12-31'))
t2m = temp_2m_be['Temperature_Air_2m_Mean_24h'] - 273.15 # K to C conversion
t2m['units'] = 'C'
t2m = t2m.rename({'lon': 'x', 'lat' : 'y'})
t2m = t2m.rio.write_crs('EPSG:4326')
world = gpd.read_file(url)
belgium = world[world['NAME'] == 'Belgium']
t2m_be = t2m.rio.clip(belgium.geometry, belgium.crs, drop = True)

# Initial Visualisation
fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

t2m_be.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(),
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01-01T01:00:00', loc = 'left')
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

plt.show()

# 1. Daily mean

temp_2m_be_daily_mean = t2m_be.resample(time='D').mean()

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

temp_2m_be_daily_mean.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01-01 (Daily Average)', loc = 'left')
ax.gridlines()
plt.show()

# 2. Monthly mean

temp_2m_be_monthly_mean = temp_2m_be_daily_mean.resample(time='ME').mean()
# Geoplot
fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

temp_2m_be_monthly_mean.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01 (Monthly Average)', loc = 'left')
ax.gridlines()
plt.show()

t2m_monthly_mean_whole = temp_2m_be_monthly_mean.mean(dim = ['x', 'y'])
time_in_strftime = t2m_monthly_mean_whole.time.dt.strftime('%Y-%m')
# Time series Plot
fig, ax = plt.subplots()
ax.plot(time_in_strftime, t2m_monthly_mean_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)
plt.show()

# Average Geo Plot (2017 - 2024)
temp_mean = temp_2m_be_monthly_mean.mean(dim='time')

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.dpi': 600, 'axes.titlesize': 16, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11})

fig = plt.figure(figsize=(11, 8), dpi=600)
ax = plt.axes(projection=ccrs.PlateCarree())

im = temp_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap= cmocean.cm.thermal, add_colorbar=False)
cbar = plt.colorbar( im, ax=ax, orientation='horizontal', pad=0.06, shrink=0.85, aspect=35)
cbar.set_label('Temperature (°C)', fontsize=12)
cbar.ax.tick_params(labelsize=11)

ax.coastlines(resolution='10m', linewidth=1.2)
ax.add_feature(cfeature.BORDERS, linewidth=0.8)
ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue', alpha=0.4)
ax.add_feature(cfeature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS, linewidth=0.4)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.6, color='gray', alpha=0.6, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 11}
gl.ylabel_style = {'size': 11}
ax.set_title('Average Air Temperature (2017–2024)', loc='center', pad=12, weight = 'bold')
ax.spines['geo'].set_linewidth(1.2)
plt.tight_layout()
plt.savefig('Mean_Temperature_2017_2024.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# 3. Monthly min

temp_2m_be_monthly_min = temp_2m_be_daily_mean.resample(time='ME').min()
# t2m_monthly_mean = temp_2m_be_monthly_mean['t2m'] - 273.15

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

temp_2m_be_monthly_min.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01 (Monthly Minimum)', loc = 'left')
ax.gridlines()
plt.show()

t2m_monthly_min_whole = temp_2m_be_monthly_min.mean(dim = ['x', 'y'])
time_in_strftime = t2m_monthly_mean_whole.time.dt.strftime('%Y-%m')

fig, ax = plt.subplots()
ax.plot(time_in_strftime, t2m_monthly_min_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)

# 4. Monthly max

temp_2m_be_monthly_max = temp_2m_be_daily_mean.resample(time='ME').max()
# t2m_monthly_mean = temp_2m_be_monthly_mean['t2m'] - 273.15

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

temp_2m_be_monthly_max.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01 (Monthly Maximum)', loc = 'left')
ax.gridlines()
plt.show()

t2m_monthly_max_whole = temp_2m_be_monthly_max.mean(dim = ['x', 'y'])
time_in_strftime = t2m_monthly_mean_whole.time.dt.strftime('%Y-%m')

fig, ax = plt.subplots()
ax.plot(time_in_strftime, t2m_monthly_max_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)

BE_Temp_df = pd.DataFrame({'YY-MM': time_in_strftime,
                        'Mean_Temp': t2m_monthly_mean_whole,
                        'Min_Temp': t2m_monthly_min_whole,
                        'Max_Temp': t2m_monthly_max_whole})

BE_Temp_df.to_csv(r'D:\MSc Thesis\Climate Data Files\BE_Temp_df.csv', index = False,)

# %% 2m Relative Humidity Preprocessing

humidity_path = r'C:\Users\r1015311\Desktop\Climate Data\Agrometerological Indicators\2m Humidity\*.nc'
humidity = xr.open_mfdataset(humidity_path, )

humidity_be = humidity.sel(lat = slice(52.5, 47.5),lon = slice(1.75, 6.9), time = slice('2017-01-01', '2024-12-31'))

rh_vars = [
    "Relative_Humidity_2m_06h",
    "Relative_Humidity_2m_09h",
    "Relative_Humidity_2m_12h",
    "Relative_Humidity_2m_15h",
    "Relative_Humidity_2m_18h"]
rh_combined = xr.concat([humidity_be[v] for v in rh_vars], dim="hour")
humidity_be["Relative_Humidity_2m_Daily_Mean"] = rh_combined.mean(dim="hour")
humidity_be_daily = humidity_be.drop_vars(rh_vars)
humidity_be_daily = humidity_be_daily.rename({'lon': 'x', 'lat' : 'y'})
humidity_be_daily = humidity_be_daily.rio.write_crs('EPSG:4326')
world = gpd.read_file(url)
belgium = world[world['NAME'] == 'Belgium']

hum_be = humidity_be_daily.rio.clip(belgium.geometry, belgium.crs, drop = True)
hum_be = hum_be['Relative_Humidity_2m_Daily_Mean']

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

hum_be.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(),
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Precipitation at 2017-01-01T01:00:00', loc = 'left')
ax.gridlines()
plt.show()

# 1. Daily mean

hum_be_daily_mean = hum_be.resample(time='D').mean()

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

hum_be_daily_mean.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01-01 (Daily Average)', loc = 'left')
ax.gridlines()
plt.show()

# 2. Monthly mean

hum_be_monthly_mean = hum_be_daily_mean.resample(time='ME').mean()

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

hum_be_monthly_mean.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Humidity (%)'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Relative Humidity at 2017-01 (Monthly Average)', loc = 'left')
ax.gridlines()
plt.show()

hum_monthly_mean_whole = hum_be_monthly_mean.mean(dim = ['x', 'y'])
time_in_strftime = hum_monthly_mean_whole.time.dt.strftime('%Y-%m')

fig, ax = plt.subplots()
ax.plot(time_in_strftime, hum_monthly_mean_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)
plt.show()

# Average Relative Humidity (2017 - 2024)
rh_mean = hum_be_monthly_mean.mean(dim="time")
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.dpi': 600, 'axes.titlesize': 16, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11})

fig = plt.figure(figsize=(11, 8), dpi=600)
ax = plt.axes(projection=ccrs.PlateCarree())

im = rh_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap= cmocean.cm.deep, add_colorbar=False)
cbar = plt.colorbar( im, ax=ax, orientation='horizontal', pad=0.06, shrink=0.85, aspect=35)
cbar.set_label('Relative Humidity (%)', fontsize=12)
cbar.ax.tick_params(labelsize=11)

ax.coastlines(resolution='10m', linewidth=1.2)
ax.add_feature(cfeature.BORDERS, linewidth=0.8)
ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue', alpha=0.4)
ax.add_feature(cfeature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS, linewidth=0.4)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.6, color='gray', alpha=0.6, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 11}
gl.ylabel_style = {'size': 11}
ax.set_title('Average Relative Humidity (2017–2024)', loc='center', pad=12, weight = 'bold')
ax.spines['geo'].set_linewidth(1.2)
plt.tight_layout()
plt.savefig('Mean_Relative_Humidity_2017_2024.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# 3. Monthly min

hum_be_monthly_min = hum_be_daily_mean.resample(time='ME').min()
# t2m_monthly_mean = temp_2m_be_monthly_mean['t2m'] - 273.15

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

hum_be_monthly_min.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01 (Monthly Minimum)', loc = 'left')
ax.gridlines()
plt.show()

hum_monthly_min_whole = hum_be_monthly_min.mean(dim = ['x', 'y'])
time_in_strftime = hum_monthly_min_whole.time.dt.strftime('%Y-%m')

fig, ax = plt.subplots()
ax.plot(time_in_strftime, hum_monthly_min_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)

# 4. Monthly max

hum_be_monthly_max = hum_be_daily_mean.resample(time='ME').max()
# t2m_monthly_mean = temp_2m_be_monthly_mean['t2m'] - 273.15

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

hum_be_monthly_max.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01 (Monthly Average)', loc = 'left')
ax.gridlines()
plt.show()

hum_monthly_max_whole = hum_be_monthly_max.mean(dim = ['x', 'y'])
time_in_strftime = hum_monthly_max_whole.time.dt.strftime('%Y-%m')

fig, ax = plt.subplots()
ax.plot(time_in_strftime, hum_monthly_max_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)

BE_Hum_df = pd.DataFrame({'YY-MM': time_in_strftime,
                        'Mean_Hum': hum_monthly_mean_whole})

BE_Hum_df.to_csv(r'D:\MSc Thesis\Climate Data Files\BE_Humi_df.csv', index = False,)
# %% 10m Windspeed Preprocessing

windspeed_path = r'C:\Users\r1015311\Desktop\Climate Data\Agrometerological Indicators\10m Wind Speed\*.nc'
windspeed = xr.open_mfdataset(windspeed_path,)

windspeed_be = windspeed.sel(lat = slice(52.5, 47.5),lon = slice(1.75, 6.9), time = slice('2017-01-01', '2024-12-31'))
ws = windspeed_be['Wind_Speed_10m_Mean_24h']
ws = ws.rename({'lon': 'x', 'lat' : 'y'})
ws = ws.rio.write_crs('EPSG:4326')
world = gpd.read_file(url)
belgium = world[world['NAME'] == 'Belgium']
ws_be = ws.rio.clip(belgium.geometry, belgium.crs, drop = True)
# Individual Visualisation
fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

ws_be.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(),
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Precipitation at 2017-01-01T01:00:00', loc = 'left')
ax.gridlines()
plt.show()

# 1. Daily mean

ws_be_daily_mean = ws_be.resample(time='D').mean()
# t2m_daily_mean = temp_2m_be_daily_mean['t2m'] - 273.15

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

ws_be_daily_mean.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Windspeed at 2017-01-01 (Daily Average)', loc = 'left')
ax.gridlines()
plt.show()

# 2. Monthly mean

ws_be_monthly_mean = ws_be_daily_mean.resample(time='ME').mean()
# t2m_monthly_mean = temp_2m_be_monthly_mean['t2m'] - 273.15

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

ws_be_monthly_mean.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Windspeed at 2017-01 (Monthly Average)', loc = 'left')
ax.gridlines()
plt.show()

ws_monthly_mean_whole = ws_be_monthly_mean.mean(dim = ['x', 'y'])
time_in_strftime = ws_monthly_mean_whole.time.dt.strftime('%Y-%m')

fig, ax = plt.subplots()
ax.plot(time_in_strftime, ws_monthly_mean_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)

# Average Windspeed (2017 - 2024)

ws_mean = ws_be_monthly_mean.mean(dim = 'time')
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.dpi': 600, 'axes.titlesize': 16, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11})

fig = plt.figure(figsize=(11, 8), dpi=600)
ax = plt.axes(projection=ccrs.PlateCarree())

im = ws_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap= cmocean.cm.haline, add_colorbar=False)
cbar = plt.colorbar( im, ax=ax, orientation='horizontal', pad=0.06, shrink=0.85, aspect=35)
cbar.set_label(r'Windspeed $\mathregular{(ms^{-1})}$', fontsize=12)
cbar.ax.tick_params(labelsize=11)

ax.coastlines(resolution='10m', linewidth=1.2)
ax.add_feature(cfeature.BORDERS, linewidth=0.8)
ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue', alpha=0.4)
ax.add_feature(cfeature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS, linewidth=0.4)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.6, color='gray', alpha=0.6, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 11}
gl.ylabel_style = {'size': 11}
ax.set_title('Average Windspeed (2017–2024)', loc='center', pad=12, weight = 'bold')
ax.spines['geo'].set_linewidth(1.2)
plt.tight_layout()
plt.savefig('Mean_Windspeed_2017_2024.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# 3. Monthly min

ws_be_monthly_min = ws_be_daily_mean.resample(time='ME').min()
# t2m_monthly_mean = temp_2m_be_monthly_mean['t2m'] - 273.15

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

ws_be_monthly_min.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01 (Monthly Minimum)', loc = 'left')
ax.gridlines()
plt.show()

ws_monthly_min_whole = ws_be_monthly_min.mean(dim = ['x', 'y'])
time_in_strftime = ws_monthly_min_whole.time.dt.strftime('%Y-%m')

fig, ax = plt.subplots()
ax.plot(time_in_strftime, ws_monthly_min_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)

# 4. Monthly max

ws_be_monthly_max = ws_be_daily_mean.resample(time='ME').max()
# t2m_monthly_mean = temp_2m_be_monthly_mean['t2m'] - 273.15

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

ws_be_monthly_max.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Windspeed at 2017-01 (Monthly Maximum)', loc = 'left')
ax.gridlines()
plt.show()

ws_monthly_max_whole = ws_be_monthly_max.mean(dim = ['x', 'y'])
time_in_strftime = ws_monthly_max_whole.time.dt.strftime('%Y-%m')

fig, ax = plt.subplots()
ax.plot(time_in_strftime, ws_monthly_max_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)

BE_Wind_df = pd.DataFrame({'YY-MM': time_in_strftime,
                        'Mean_Prec': ws_monthly_mean_whole,
                        'Min_Prec': ws_monthly_min_whole,
                        'Max_Prec': ws_monthly_max_whole})

BE_Wind_df.to_csv(r'D:\MSc Thesis\Climate Data Files\BE_Wind_df.csv', index = False,)

# %% Precipitation Flux Preprocessing

precipitation_path = r'C:\Users\r1015311\Desktop\Climate Data\Agrometerological Indicators\Precipitation Flux\*.nc'
precipitation = xr.open_mfdataset(precipitation_path, )

precipitation_be = precipitation.sel(lat = slice(52.5, 47.5),lon = slice(1.75, 6.9), time = slice('2017-01-01', '2024-12-31'))
tp = precipitation_be['Precipitation_Flux']
tp = tp.rename({'lon': 'x', 'lat' : 'y'})
tp = tp.rio.write_crs('EPSG:4326')
world = gpd.read_file(url)
belgium = world[world['NAME'] == 'Belgium']
tp_be = tp.rio.clip(belgium.geometry, belgium.crs, drop = True)
# Initial Visaulisation
fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

tp_be.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(),
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Precipitation at 2017-01-01T01:00:00', loc = 'left')
ax.gridlines()
plt.show()

# 1. Daily mean

tp_be_daily_mean = tp_be.resample(time='D').mean()

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

tp_be_daily_mean.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01-01 (Daily Average)', loc = 'left')
ax.gridlines()
plt.show()

# 2. Monthly mean

tp_be_monthly_mean = tp_be_daily_mean.resample(time='ME').mean()
# Geo plot
fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

tp_be_monthly_mean.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01 (Monthly Average)', loc = 'left')
ax.gridlines()
plt.show()

tp_monthly_mean_whole = tp_be_monthly_mean.mean(dim = ['x', 'y'])
time_in_strftime = tp_monthly_mean_whole.time.dt.strftime('%Y-%m')
# Time series plot
fig, ax = plt.subplots()
ax.plot(time_in_strftime, tp_monthly_mean_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)

# Average Precipitation (2017 - 2024)
tp_mean = tp_be_monthly_mean.mean(dim="time")

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.dpi': 600, 'axes.titlesize': 16, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11})

fig = plt.figure(figsize=(11, 8), dpi=600)
ax = plt.axes(projection=ccrs.PlateCarree())

im = tp_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap= cmocean.cm.rain, add_colorbar=False)
cbar = plt.colorbar( im, ax=ax, orientation='horizontal', pad=0.06, shrink=0.85, aspect=35)
cbar.set_label(r'Precipitation $\mathregular{(mmday^{-1})}$', fontsize=12)
cbar.ax.tick_params(labelsize=11)

ax.coastlines(resolution='10m', linewidth=1.2)
ax.add_feature(cfeature.BORDERS, linewidth=0.8)
ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue', alpha=0.4)
ax.add_feature(cfeature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS, linewidth=0.4)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.6, color='gray', alpha=0.6, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 11}
gl.ylabel_style = {'size': 11}
ax.set_title('Average Precipitation (2017–2024)', loc='center', pad=12, weight = 'bold')
ax.spines['geo'].set_linewidth(1.2)
plt.tight_layout()
plt.savefig('Mean_Precipitation_2017_2024.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# 3. Monthly min

tp_be_monthly_min = tp_be_daily_mean.resample(time='ME').min()
# t2m_monthly_mean = temp_2m_be_monthly_mean['t2m'] - 273.15

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

tp_be_monthly_min.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01 (Monthly Minimum)', loc = 'left')
ax.gridlines()
plt.show()

tp_monthly_min_whole = tp_be_monthly_min.mean(dim = ['x', 'y'])
time_in_strftime = tp_monthly_min_whole.time.dt.strftime('%Y-%m')

fig, ax = plt.subplots()
ax.plot(time_in_strftime, tp_monthly_min_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)

# 4. Monthly max

tp_be_monthly_max = tp_be_daily_mean.resample(time='ME').max()
# t2m_monthly_mean = temp_2m_be_monthly_mean['t2m'] - 273.15

fig, ax = plt.subplots(subplot_kw = {'projection' : ccrs.PlateCarree()}, figsize = (10, 7))

tp_be_monthly_max.isel(time = 0).plot(ax = ax, transform = ccrs.PlateCarree(), 
                              cmap = 'coolwarm', cbar_kwargs = {'label' : 'Temperature'})
ax.set_title('')
ax.coastlines(resolution = '10m', lw = 1.2)
ax.add_feature(cfeature.BORDERS, lw = 1)
ax.add_feature(cfeature.OCEAN, facecolor = 'lightskyblue', alpha = 0.4)
ax.set_title('Temperature at 2017-01 (Monthly Maximum)', loc = 'left')
ax.gridlines()
plt.show()

tp_monthly_max_whole = tp_be_monthly_max.mean(dim = ['x', 'y'])
time_in_strftime = tp_monthly_max_whole.time.dt.strftime('%Y-%m')

fig, ax = plt.subplots()
ax.plot(time_in_strftime, tp_monthly_max_whole)
ax.set_xticks(time_in_strftime[::6])
ax.tick_params('x', rotation = 90)
ax.grid('--', c = 'grey', alpha = 0.3)

BE_Prec_df = pd.DataFrame({'YY-MM': time_in_strftime,
                        'Mean_Prec': tp_monthly_mean_whole,
                        'Min_Prec': tp_monthly_min_whole,
                        'Max_Prec': tp_monthly_max_whole})

BE_Prec_df.to_csv(r'D:\MSc Thesis\Climate Data Files\BE_Prec_df.csv', index = False,)

# %% END OF SCRIPT