# Copyright (c) Tao Han: hantao10200@gmail.com. All rights reserved.
import xarray as xr
import argparse
import matplotlib.pyplot as plt
import imageio
import io
import os
from datetime import datetime, timedelta
import numpy as np 
from mpl_toolkits.basemap import Basemap
import torch.nn.functional as F
from scipy.ndimage import zoom
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime as dt
import cartopy.mpl.ticker as cticker
from scipy.ndimage import uniform_filter,gaussian_filter 
from matplotlib.markers import MarkerStyle
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
from mpl_toolkits.basemap import Basemap

fontname = 'Times New Roman'

color_bar =  {'u':'viridis', 'v': 'YlGnBu', 't':'Spectral_r', 'q':'PuBu', 'r':'YlOrBr', 'z':'coolwarm',   'msl':'coolwarm',
              'u10':'viridis', 'v10': 'YlGnBu', 'u100':'viridis', 'v100': 'YlGnBu', 't2m':'Spectral_r',
              }
unit = {'z': '$m^2/s^2$', 'q': 'kg/kg',  'u':'m/s', 'v':'m/s', 'w':'m/s', 'r':'%',  't':'K',  'msl':'Pa',  
        'u10':'m/s', 'v10': 'm/s', 'u100':'m/s', 'v100': 'm/s', 't2m':'K'}

def fig_to_image(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = imageio.imread(buf)

    return image

def compress_gif(input_file, output_file, optimization_level=3):
    reader = imageio.get_reader(input_file)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_file, fps=fps, optimize=optimization_level)

    for frame in reader:
        writer.append_data(frame)

    reader.close()
    writer.close()
    
def global_set(ax,img_extent):
    ax.coastlines()
    ax.set_extent(img_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    # ax.set_xticks(np.arange(img_extent[0],img_extent[1], 5), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(img_extent[2],img_extent[3], 5), crs=ccrs.PlateCarree())
    
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.grid(False)
    
def plot_pressure_demo_gif(initial_timestamp:str, 
                           steps: int, 
                           plot_variable:dict,
                           output_root: str):
    
    images = {}     # creat a dict to store images
    for step in range(steps): 
        new_dt = datetime.fromisoformat(initial_timestamp) + timedelta(hours=6*(step+1))
        fc_timestamp = new_dt.isoformat()
        
        # read pressure_level forecast data
        forecast_dataset = xr.open_dataset(
                                    f'{output_root}/{initial_timestamp}/{fc_timestamp}_pressure.nc',
                                    engine='netcdf4',  
                                    )

        # import pdb
        # pdb.set_trace()
        forecast_single = xr.open_dataset(
                                    f'{output_root}/{initial_timestamp}/{fc_timestamp}_surface.nc',
                                    engine='netcdf4',  
                                    )
            
        vnames = forecast_dataset.data_vars.keys()
        fontsize = 24
        print(f'process step {step}')
        for vname in vnames:
            fig_list = []  # Create a list to store the figures for each vname
            forecast_data = forecast_dataset[vname]
            heights = forecast_data.isobaricInhPa.data
            if vname not in plot_variable.keys():
                continue
            if vname not in images.keys():
                images.update({vname:[]})
            for height in heights:
                v_fullname= f'{vname}_{int(height)}'
                if height not in plot_variable[vname]:
                    continue
                fc_data= forecast_data.sel(isobaricInhPa=height).values.squeeze()  #

                input_shape = fc_data.shape
                
                # print(f'{v_fullname}:{fc_timestamp}:{np.sqrt(np.square(fc_data-real_data).mean())}')
                aspect_ratio = fc_data.shape[1] / fc_data.shape[0]  # Calculate the aspect ratio of the image
                fig_width = 6  # Set the desired width of the figure
                fig_height = fig_width / aspect_ratio # Calculate the height of the figure based on the aspect rat
                
                leftlon, rightlon, lowerlat, upperlat = (110,130, 10,30)
                img_extent = [leftlon, rightlon, lowerlat, upperlat]
                
                fig, axes = plt.subplots(1, 1, 
                                        figsize=(fig_width*2, fig_height*1.5),
                                        subplot_kw={'projection': ccrs.Robinson(central_longitude=180)}, #ccrs.Robinson() ccrs.PlateCarree(
                                        gridspec_kw={
                                        # 'width_ratios': [2, 1],
                                        # 'height_ratios': [1,1], 
                                        'wspace': 0.05,  
                                        'hspace': 0.06  
                                    })
                
                ax1 = axes
                global_set(ax1)
                # Plot forecast image
                im1 = ax1.imshow(fc_data, cmap=color_bar.get(vname.split('_')[0], None), extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
                ax1.set_title(f'Forecast: FengWu-GHR (Hres 0.09°)',fontsize=fontsize,)
                ax1.set_axis_off()
                
                # Add a general title
                fig.suptitle(f"{v_fullname}: Initialized at: {initial_timestamp}, forecasting at: {fc_timestamp}",  fontsize=fontsize+2, y=0.95)
                
                fig.subplots_adjust(top=0.99, bottom=0.0, left=0.01, right=0.99)  # Manually adjust the spacing between subplots

                fig_list.append(fig_to_image(fig))
                plt.close()

            # Stack the images vertically using numpy.vstack()
            stacked_image = np.vstack(np.array(fig_list))
            images[vname].append(stacked_image)
                    
    for key, image_list in images.items():
        os.makedirs(f'./demos/',exist_ok=True)
        save_path =f'./demos/{key}_forecast_only.gif'
        imageio.mimsave(save_path, image_list, fps=1)
        # compress_gif(save_path,save_path)
        

def plot_surface_demo_gif(initial_timestamp:str, 
                          steps: int, 
                          plot_variable:dict, 
                          output_root: str):
    
            
    leftlon, rightlon, lowerlat, upperlat = (118,138, 20,40)
    img_extent = [leftlon, rightlon, lowerlat, upperlat]
    images = []     # creat a list to store images
    h_num, w_num =steps//2, 4
    fig, axes = plt.subplots(h_num, w_num, 
                        figsize=(2*w_num, 2.1*h_num),
                        subplot_kw={'projection': ccrs.Robinson(central_longitude=180)}, #ccrs.Robinson() ccrs.PlateCarree(
                        gridspec_kw={
                        # 'width_ratios': [2, 1],
                        # 'height_ratios': [1,1], 
                        'wspace': 0.15,  
                        'hspace': 0.15  
                    })
    axes = axes.flatten()
    fontsize = 12
    for step in range(steps): 
        print(f'step:{step}')
        new_dt = datetime.fromisoformat(initial_timestamp) + timedelta(hours=6*(step+1))
        fc_timestamp = new_dt.isoformat()
        # read surface forecast data
        forecast_dataset = xr.open_dataset(
                                    f'{output_root}/{initial_timestamp}/{fc_timestamp}_surface.nc',
                                    engine='netcdf4',  
                                    )
        vnames = forecast_dataset.data_vars.keys()

        vnames  = ['u10', 'v10', 'msl']
        u10 = forecast_dataset['u10'].values.squeeze() 
        v10 = forecast_dataset['v10'].values.squeeze() 
        msl = forecast_dataset['msl'].values.squeeze() 
        ws = np.sqrt(u10**2+v10**2)
        
        fig_list = []  # Create a list to store the figures for each vname
        
        ax1 = axes[step*2]
        global_set(ax1, img_extent)

        im1 = ax1.imshow(ws, cmap='viridis', extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        ax1.set_title(f'ws :{fc_timestamp}',fontsize=fontsize,)
        ax1.set_axis_off()
        
        ax1 = axes[step*2+1]
        global_set(ax1, img_extent)

        im1 = ax1.imshow(msl, cmap='coolwarm', extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        ax1.set_title(f'msl:{fc_timestamp}',fontsize=fontsize,)
        ax1.set_axis_off()
        
        fig.subplots_adjust(top=0.99, bottom=0.0, left=0.01, right=0.99)  # Manually adjust the spacing between subplots
        
        fig_list.append(fig_to_image(fig))
    plt.savefig(f'./demos/surface_forecast.png')

        # Stack the images vertically using numpy.vstack()
        # stacked_image = np.vstack(np.array(fig_list))
        # images.append(stacked_image)
                    
   
    # generate GIF image
    # os.makedirs(f'./demos/',exist_ok=True)
    # save_path = f'./demos/surface_forecast.gif'
    
    # imageio.mimsave(save_path, images, fps=2)
    # compress_gif(save_path,save_path)
from scipy.ndimage import gaussian_filter    
def plot_surface_single_image(initial_timestamp:str, 
                          steps: int, 
                          plot_variable:dict, 
                          output_root: str):
    leftlon, rightlon, lowerlat, upperlat =   (90,150, 10,50)  # chinese east sea (-180, 180, -90, 90)  #
    img_extent = [leftlon, rightlon, lowerlat, upperlat]

    fontsize = 20
    sample = 36  
    fig_ws_list = []  # Create a list to store the figures for each vname
    fig_msl_list = []
    for step in range(steps): 
        print(f'step:{step}')
        fig, axes = plt.subplots(1, 1, 
                    figsize=(10, 5),
                    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, #ccrs.Robinson() ccrs.PlateCarree(
                    gridspec_kw={
                    # 'width_ratios': [2, 1],
                    # 'height_ratios': [1,1], 
                    'wspace': 0.1,  
                    'hspace': 0.1  
                })
        new_dt = datetime.fromisoformat(initial_timestamp) + timedelta(hours=6*(step+1)) 
        bj_fc_timestamp = new_dt + timedelta(hours=8)
        fc_timestamp = new_dt.isoformat()
        bj_fc_fc_timestamp = bj_fc_timestamp.isoformat()
        # read surface forecast data
        forecast_dataset = xr.open_dataset(
                                    f'{output_root}/{initial_timestamp}/{fc_timestamp}_surface.nc',
                                    engine='netcdf4',  
                                    )
        lat = forecast_dataset.latitude
        lon = forecast_dataset.longitude
        
        vnames = forecast_dataset.data_vars.keys()
        vnames  = ['u10', 'v10', 'msl']
        
        

        ax1 = axes
        global_set(ax1, img_extent)

        u10 = forecast_dataset['u10'].values.squeeze() 

        v10 = forecast_dataset['v10'].values.squeeze()
        ws = np.sqrt(u10**2+v10**2)
        # ws = gaussian_filter(ws, sigma=3)
        # import pdb
        # pdb.set_trace()
        q = ax1.quiver(lon[::sample],lat[::sample], u10[::sample,::sample], v10[::sample,::sample], \
                       transform=ccrs.PlateCarree(), scale=100, color='r',width=0.003,headwidth=3,headlength=6)
        ax1.quiverkey(q, 1, 1.02, 10, '10 m/s', labelpos='E',coordinates='axes', rasterized=True)
        # c = ax1.contourf(lon,lat, ws, cmap='viridis', levels=30, extend='both', transform=ccrs.PlateCarree(), zorder=0)      
        c = ax1.imshow(ws, cmap='viridis', extent=[0, 360, -90, 90], transform=ccrs.PlateCarree(), zorder=0)
        cbar = fig.colorbar(c, drawedges=False, ax=ax1, location='right', shrink=0.7, pad=0.01, spacing='uniform')  
        cbar.ax.tick_params(labelsize=10)  # 设置色标尺标签大小 
        
        ax1.set_title(f'Wind Speed:{fc_timestamp} (m/s)',fontsize=fontsize)
        ax1.set_axis_off()
        
        fig.subplots_adjust(top=0.99, bottom=0.0, left=0.01, right=0.99)  # Manually adjust the spacing between subplots
        
        os.makedirs(f'./demos/Bebinca/',exist_ok=True)
        plt.savefig(f'./demos/Bebinca/{step}_ws.png', dpi=200, bbox_inches='tight')
        fig_ws_list.append(fig_to_image(fig))
        
        
        
        fig, axes = plt.subplots(1, 1, 
            figsize=(8.5, 8.5),
            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, #ccrs.Robinson() ccrs.PlateCarree(
            gridspec_kw={
            # 'width_ratios': [2, 1],
            # 'height_ratios': [1,1], 
            'wspace': 0.1,  
            'hspace': 0.1  
        })
        ax1 = axes
        global_set(ax1, img_extent)
        
                
        msl = forecast_dataset['msl'].values.squeeze() /100
        # msl = gaussian_filter(msl, sigma=3)

        X,Y = np.meshgrid(lon,lat)
        C =  ax1.contour(X,Y, msl, 16,  colors='green',  linewidths=0.4, transform=ccrs.PlateCarree())
        # filled = ax.contourf(X, Y, msl, cmap='coolwarm', levels=30, transform=ccrs.PlateCarree(),vmin=vmin, vmax=vmax)
        im = ax1.imshow(msl, cmap='coolwarm', extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        plt.clabel(C, inline=True, fontsize=10, colors="k") #显示等值线上数
        cbar = fig.colorbar(im, drawedges=False, ax=ax1, location='right', shrink=0.7, pad=0.01, spacing='uniform') # label='Sea Level Pressure (hPa)  
        cbar.ax.tick_params(labelsize=10)  # 设置色标尺标签大小 
        
        ax1.set_title(f'Sea-level Pressure:{fc_timestamp} (hPa)',fontsize=fontsize)
        ax1.set_axis_off()
        
        fig.subplots_adjust(top=0.99, bottom=0.0, left=0.01, right=0.99)  # Manually adjust the spacing between subplots
        
        os.makedirs(f'./demos/Bebinca/',exist_ok=True)
        plt.savefig(f'./demos/Bebinca/{step}_msl.png', dpi=200, bbox_inches='tight')
     
        fig_msl_list.append(fig_to_image(fig))
        # Stack the images vertically using numpy.vstack()

    # generate GIF image
    os.makedirs(f'./demos/',exist_ok=True)
    save_path = f'./demos/surface_forecast_ws.gif'
    
    imageio.mimsave(save_path, fig_ws_list, fps=2)
    
    save_path = f'./demos/surface_forecast_msl.gif'
    imageio.mimsave(save_path, fig_msl_list, fps=2)
    # compress_gif(save_path,save_path)
    
def fig_to_gif(step):
    fig_ws_list = []
    fig_msl_list = []
    for i in range(step):
        im1_path = f'./demos/Bebinca/{i}_ws.png'
        im2_path = f'./demos/Bebinca/{i}_msl.png'
        image = imageio.imread(im1_path)
        fig_ws_list.append(image)
        image = imageio.imread(im2_path)
        fig_msl_list.append(image)
    os.makedirs(f'./demos/',exist_ok=True)
    save_path = f'./demos/surface_forecast_ws.gif'
    
    imageio.mimsave(save_path, fig_ws_list, fps=1)
    
    save_path = f'./demos/surface_forecast_msl.gif'
    imageio.mimsave(save_path, fig_msl_list, fps=1)
    
def parse_args():
    parser = argparse.ArgumentParser(description='FengWu-GHR forecast demo')
    
    parser.add_argument('--timestamp',
                        default='2024-07-08T18:00:00',
                        type=str,
                        help='The timestamp of the initial field.')

    parser.add_argument('--dataset',
                        default=None,
                        type=str,
                        help='The datasource of the initial field. Both ERA5 and analysis from EC are supported.'
                        )
    parser.add_argument('--inference_steps',
                        default=1,
                        type=int,
                        help='The forecasting lead time steps.'
                        )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    output_root = os.path.join('./data/output', args.dataset)
    # plot_pressure_demo_gif(initial_timestamp = args.timestamp, 
    #                 steps=args.inference_steps, #cfg.inference_steps, 
    #                 plot_variable = {'z':[850,500],
    #                                 'q':[850,500],
    #                                 'u':[850,500],
    #                                 'v':[850,500],
    #                                 't':[850,500],
    #                                 },
    #                 output_root = output_root
    #                 )
    
    # plot_surface_demo_gif(
    #                 initial_timestamp=args.timestamp, 
    #                 steps=args.inference_steps, 
    #                 plot_variable=['sp','v10','v100', 't2m','tp6h', 'msl'],
    #                 output_root = output_root
    # #                             )
    plot_surface_single_image(
                    initial_timestamp=args.timestamp, 
                    steps=args.inference_steps, 
                    plot_variable=['sp','v10','v100', 't2m','tp6h', 'msl'],
                    output_root = output_root
                                )
    fig_to_gif(30)