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


color_bar =  {'u':'viridis', 'v': 'YlGnBu', 't':'Spectral_r', 'q':'PuBu', 'r':'YlOrBr', 'z':'coolwarm',   'msl':'coolwarm',
              'u10':'viridis', 'v10': 'YlGnBu', 'u100':'viridis', 'v100': 'YlGnBu', 't2m':'Spectral_r',
              }
unit = {'z': '$m^2/s^2$', 'q': 'kg/kg',  'u':'m/s', 'v':'m/s', 'w':'m/s', 'r':'%',  't':'K',  'msl':'Pa',  
        'u10':'m/s', 'v10': 'm/s', 'u100':'m/s', 'v100': 'm/s', 't2m':'K'}

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=50)
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
    
def global_set(ax):
    f_size = 12
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    ax.tick_params(labelsize=f_size)
    # ax.gridlines(linestyle='--')
    ax.set_global()
    
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

        # read reanalysis data
        real_dataset = xr.open_dataset(
            f'./data/input/era5/{fc_timestamp[:4]}/{fc_timestamp}.grib', 
            engine='cfgrib',
            backend_kwargs={'indexpath': ''}
            )
        
        # import pdb
        # pdb.set_trace()
        # forecast_single = xr.open_dataset(
        #                             f'./data/output/{initial_timestamp}/{fc_timestamp}_surface.nc',
        #                             engine='netcdf4',  
        #                             )
        # real_single = xr.open_dataset(
        #     f'./data/input/era5/{fc_timestamp[:4]}/{fc_timestamp}_single.nc', 
        #     #engine='cfgrib',
        #     #backend_kwargs={'indexpath': ''}
        #     )
 
            
            
        vnames = forecast_dataset.data_vars.keys()
        fontsize = 24
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
                real_data = real_dataset[vname].sel(isobaricInhPa=height).values.squeeze()
                

                input_shape = fc_data.shape
                real_data = zoom(real_data, 
                        ( input_shape[0] / real_data.shape[0],
                          input_shape[1] / real_data.shape[1]),
                        order=1)  #
        
                
                print(f'{v_fullname}:{fc_timestamp}:{np.sqrt(np.square(fc_data-real_data).mean())}')
                aspect_ratio = fc_data.shape[1] / fc_data.shape[0]  # Calculate the aspect ratio of the image
                fig_width = 12  # Set the desired width of the figure
                fig_height = fig_width / aspect_ratio # Calculate the height of the figure based on the aspect rat
                fig, axes = plt.subplots(1, 2, 
                                        figsize=(fig_width*2, fig_height*1.5),
                                        subplot_kw={'projection': ccrs.Robinson(central_longitude=180)}, #ccrs.Robinson() ccrs.PlateCarree()
                                        gridspec_kw={
                                        # 'width_ratios': [2, 1],
                                        # 'height_ratios': [1,1], 
                                        'wspace': 0.05,  
                                        'hspace': 0.06  
                                    })
                
                ax1 = axes[0]
                ax2 = axes[1]
                global_set(ax1)
                global_set(ax2)
                # Plot forecast image
                im1 = ax1.imshow(fc_data, cmap=color_bar.get(vname.split('_')[0], None), extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
                ax1.set_title(f'Forecast: FengWu-GHR (Hres 0.09°)',fontsize=fontsize,)
                ax1.set_axis_off()
                

                # Plot real image
                im2 = ax2.imshow(real_data,cmap=color_bar.get(vname.split('_')[0], None), extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
                ax2.set_title(f'Truth: ERA5 (upsample to 0.09°)', fontsize=fontsize,)
                ax2.set_axis_off()


                # Create a new set of axes for the colorbar
                cax = fig.add_axes([0.25, 0.1, 0.5, 0.05])
                cbar = fig.colorbar(im1, cax=cax, orientation='horizontal')
                cbar.set_label(unit.get(vname.split('_')[0], None), fontsize=fontsize)
                cbar.ax.tick_params(labelsize=fontsize)
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
        save_path =f'./demos/{key}_forecast_vs_real.gif'
        imageio.mimsave(save_path, image_list, fps=1)
        # compress_gif(save_path,save_path)
        
def plot_surface_demo_gif(initial_timestamp:str, 
                          steps: int, 
                          plot_variable:dict, 
                          output_root: str):
    

    images = []     # creat a list to store images
    for step in range(steps): 
        new_dt = datetime.fromisoformat(initial_timestamp) + timedelta(hours=6*(step+1))
        fc_timestamp = new_dt.isoformat()
        # read surface forecast data
        forecast_dataset = xr.open_dataset(
                                    f'./data/output/{initial_timestamp}/{fc_timestamp}_surface.nc',
                                    engine='netcdf4',  
                                    )
        vnames = forecast_dataset.data_vars.keys()

        # variable = 't2m'
        # forecast_dataset[variable].plot()
        # plt.show()
        # plt.savefig('./data/demos/fc_t2m.png')


        fontsize = 24
        fig_list = []  # Create a list to store the figures for each vname
        for vname in vnames:
            # read reanalysis data
            # real_data = np.load(
            #     f'./data/input/era5/{fc_timestamp}/{vname}.npy', 
            #     )
            real_dataset = xr.open_dataset(
                f'./data/input/era5/{fc_timestamp}.grib', 
                #engine='cfgrib',
                #backend_kwargs={'indexpath': ''}
                )
            # forecast_dataset[variable].plot()
            # plt.show()
            # plt.savefig('./data/demos/real_t2m.png')
            
            real_data = real_dataset[vname].values.squeeze()
            
            from scipy.ndimage import zoom
            input_shape = (2001, 4000) 
            real_data = zoom(real_data, 
                         (input_shape[0] / real_data.shape[0],
                          input_shape[1] / real_data.shape[1]),
                        order=1)  #
        
            # real_data = torch.from_numpy(real_data)[None, None, :, :]
            # real_data = F.interpolate(real_data, size=input_shape, mode='bicubic')
            # real_data = real_data.squeeze().squeeze()  
            # real_data = real_data.numpy()  
            
            fc_data = forecast_dataset[vname].values.squeeze() 
            if vname not in plot_variable:
                continue

            key= f'{vname}'
 

            print(f'{key}:{fc_timestamp}:{np.sqrt(np.square(fc_data-real_data).mean())}')
            aspect_ratio = fc_data.shape[1] / fc_data.shape[0]  # Calculate the aspect ratio of the image
            fig_width = 12  # Set the desired width of the figure
            fig_height = fig_width / aspect_ratio # Calculate the height of the figure based on the aspect rat
            fig, axes = plt.subplots(1, 2, figsize=(fig_width*2, fig_height*1.35))
            
            ax1 = axes[0]
            ax2 = axes[1]
            map1 = Basemap(ax=ax1)
            map2 = Basemap(ax=ax2)

            # Plot forecast image
            im1 = map1.imshow(fc_data)
            ax1.set_title(f'Forecast: {fc_timestamp}',fontsize=fontsize,)
            ax1.set_axis_off()
            map1.drawcoastlines()
            map1.drawcountries()

            # Plot real image
            im2 = map2.imshow(real_data)
            ax2.set_title(f'Real: {fc_timestamp}', fontsize=fontsize,)
            ax2.set_axis_off()
            map2.drawcoastlines()
            map2.drawcountries()

            # Create a new set of axes for the colorbar
            cax = fig.add_axes([0.25, 0.1, 0.5, 0.05])
            cbar = fig.colorbar(im1, cax=cax, orientation='horizontal')
            cbar.set_label(key, fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)
            # Add a general title
            fig.suptitle(f"The initial field timestamp is {initial_timestamp}", fontsize=fontsize+4, y=0.95)
            
            fig.subplots_adjust(top=0.99, bottom=0.0, left=0.01, right=0.99)  # Manually adjust the spacing between subplots
            
            fig_list.append(fig_to_image(fig))
            plt.close()

        # Stack the images vertically using numpy.vstack()
        stacked_image = np.vstack(np.array(fig_list))
        images.append(stacked_image)
                    
   
    # generate GIF image
    os.makedirs(f'./demos/',exist_ok=True)
    save_path = f'./demos/surface_forecast_vs_real.gif'
    imageio.mimsave(save_path, images, fps=2)
    # compress_gif(save_path,save_path)
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
    plot_pressure_demo_gif(initial_timestamp = args.timestamp, 
                    steps=args.inference_steps, #cfg.inference_steps, 
                    plot_variable = {'z':[850,500],
                                    'q':[850,500],
                                    'u':[850,500],
                                    'v':[850,500],
                                    't':[850,500],
                                    },
                    output_root = output_root
                    )
    
    # plot_surface_demo_gif(
    #                 initial_timestamp=args.timestamp, 
    #                 steps=args.inference_steps, 
    #                 plot_variable=['sp','v10','v100', 't2m','tp6h', 'msl'],
    #                 output_root = output_root
    #                             )
