# input: Deep Graph --> g 

# feature functions, will be applied to each g_id
feature_funcs = {'daily_mag': [np.sum],
                 'latitude': np.min,
                 'longitude': np.min}

gv = g.partition_nodes('g_id', feature_funcs)
gv.rename(columns={'latitude_amin': 'lat',
                   'longitude_amin': 'lon'}, inplace=True)

cols = {'n_nodes': gv.n_nodes,
        'HWMId': gv.daily_mag_sum,}

for name, col in cols.items():

    # for easy filtering, we create a new DeepGraph instance for 
    # each component
    gt = dg.DeepGraph(gv)

    # configure map projection
    kwds_basemap = {'llcrnrlon': vt.longitude.min() - 1,
                    'urcrnrlon': vt.longitude.max() + 1,
                    'llcrnrlat': vt.latitude.min() - 1,
                    'urcrnrlat': vt.latitude.max() + 1}
    
    # configure scatter plots
    kwds_scatter = {'s': 3400,
                    'marker': 's',
                    'c': col.values,
                    'cmap': 'viridis_r',
                    'alpha': .5,
                    'edgecolors': 'none'}

    # create scatter plot on map
    obj = gt.plot_map(lon='lon', lat='lat',
                      kwds_basemap=kwds_basemap,
                      kwds_scatter=kwds_scatter)

    # configure plots
    obj['m'].drawcoastlines(linewidth=.8)
    obj['m'].drawparallels(range(-90, 90, 2), linewidth=.2)
    obj['m'].drawmeridians(range(0, 360, 2), linewidth=.2)
    obj['m'].drawcountries()
    obj['ax'].set_title(name)
    
    # colorbar
    cb = obj['fig'].colorbar(obj['pc'], fraction=.022, pad=.02)
    cb.set_label('{}'.format(name), fontsize=15)   
    
    # save the images
