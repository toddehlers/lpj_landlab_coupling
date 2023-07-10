import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import os
from lpjguesstools import plotting
import sys

import matplotlib

font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

infile = sys.argv[1]

ds = xr.open_dataset(infile)
ds = ds.sel(ni=slice(4,95), nj=slice(4,95)) #.mean(dim='time', skipna=False)

# and create custom cmap (my_map)
interval = np.linspace(0.1, 0.7)
colors = plt.cm.terrain(interval)
ele_cmap = LinearSegmentedColormap.from_list('name', colors)


var = 'elevation'
fig, ax = plt.subplots(figsize=(4,3))
ds['topographic__elevation'].mean(dim='nt', skipna=False).plot(ax=ax, cmap=ele_cmap, vmin=0, vmax=800,
        cbar_kwargs={'label': 'Elevation [m]'})
ds['topographic__elevation'].mean(dim='nt', skipna=False).plot.contour(ax=ax, 
                                                                         levels=[200,400,600], 
                                                                         colors=['k'],
                                                                         linestyles = 'dashed',
                                                                         linewidths=1.)
ax.set_aspect('equal')
plt.xlabel('')
plt.ylabel('')
plt.savefig('maps/' + os.path.basename(infile)[:-3]+ '_' + var + '.png')


var = 'vegetation'
fig, ax = plt.subplots(figsize=(4,3))

FPC_LIM = 30

(ds['vegetation__density']*100).mean(dim='nt', skipna=False).plot(ax=ax, cmap='YlGn', vmin=0, vmax=FPC_LIM,
        cbar_kwargs={'label': 'Veg. Cover [%]'})
ds['topographic__elevation'].mean(dim='nt', skipna=False).plot.contour(ax=ax, 
                                                                         levels=[200,400,600], 
                                                                         colors=['k'],
                                                                         linestyles = 'dashed',
                                                                         linewidths=1.)
ax.set_aspect('equal')
plt.xlabel('')
plt.ylabel('')
plt.savefig('maps/' + os.path.basename(infile)[:-3]+ '_' + var + '.png')

var = 'erosion'
fig, ax = plt.subplots(figsize=(4,3))

(ds['erosion__rate']*1000).mean(dim='nt', skipna=False).plot(ax=ax, cmap='plasma', vmin=0, vmax=0.5,
        cbar_kwargs={'label': 'Erosion [mm/yr]'})
ds['topographic__elevation'].mean(dim='nt', skipna=False).plot.contour(ax=ax, 
                                                                         levels=[200,400,600], 
                                                                         colors=['k'],
                                                                         linestyles = 'dashed',
                                                                         linewidths=1.)
ax.set_aspect('equal')
plt.xlabel('')
plt.ylabel('')
plt.savefig('maps/' + os.path.basename(infile)[:-3]+ '_' + var + '.png')

lfids = ds['landform__ID'].values
unique_ids = np.unique(lfids.ravel())

D={}
for i, uid in enumerate(unique_ids):
    D[uid] = i

vals = np.linspace(0,1,256)
np.random.shuffle(vals)
random_cmap = plt.cm.colors.ListedColormap(plt.cm.rainbow(vals))


new_data = np.vectorize(D.get)(lfids)
ds['landform__ID'][:] = new_data
lfid_data = ds['landform__ID'].median(dim='nt', skipna=False) 

var = 'landforms'
fig, ax = plt.subplots(figsize=(3.5,3))

lfid_data.plot(ax=ax, cmap='rainbow', add_colorbar=False) #random_cmap) # discrete_cmap(len(unique_ids), 'rainbow'))
ds['topographic__elevation'].mean(dim='nt', skipna=False).plot.contour(ax=ax, 
                                                                         levels=[200,400,600], 
                                                                         colors=['k'],
                                                                         linestyles = 'dashed',
                                                                         linewidths=1.)

ax.set_aspect('equal')
plt.xlabel('')
plt.ylabel('')
                                                                         
plt.savefig('maps/' + os.path.basename(infile)[:-3]+ '_' + var + '.png')


#map = plotting.Map(ds)



