# Copyright (c) Tao Han: hantao10200@gmail.com. All rights reserved.
import xarray as xr
import matplotlib.pyplot as plt
import imageio
import io
import os
from datetime import datetime, timedelta
import numpy as np 
from mpl_toolkits.basemap import Basemap

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

def plot_pressure_demo_gif(initial_timestamp:str, steps: int, plot_variable:dict):
    
    images = {}     # creat a dict to store images
    for step in range(steps): 
        new_dt = datetime.fromisoformat(initial_timestamp) + timedelta(hours=6*(step+1))
        fc_timestamp = new_dt.isoformat()
        
        # read pressure_level forecast data
        forecast_dataset = xr.open_dataset(
                                    f'./data/output/{initial_timestamp}/{fc_timestamp}_pressure.nc',
                                    engine='netcdf4',  
                                    )

        # read reanalysis data
        real_dataset = xr.open_dataset(
            f'./data/input/era5/{fc_timestamp}.grib', 
            engine='cfgrib',
            backend_kwargs={'indexpath': ''}
            )
        
        # import pdb
        # pdb.set_trace()

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
                key= f'{vname}_{int(height)}'
                if height not in plot_variable[vname]:
                    continue
                fc_data= forecast_data.sel(isobaricInhPa=height).values.squeeze()  # 绘制预测和真实的大气场图像
                real_data = real_dataset[vname].sel(isobaricInhPa=height).values.squeeze()

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
            # import pdb
            # pdb.set_trace()   
           
            # Stack the images vertically using numpy.vstack()
            stacked_image = np.vstack(np.array(fig_list))
            images[vname].append(stacked_image)
                    
    for key, image_list in images.items():
    # 生成 GIF 图片
        os.makedirs(f'./data/demos/',exist_ok=True)
        save_path =f'./data/demos/{key}_forecast_vs_real.gif'
        imageio.mimsave(save_path, image_list, fps=2)
        # compress_gif(save_path,save_path)
        
def plot_surface_demo_gif(initial_timestamp:str, steps: int, plot_variable:list):
    

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
        # import pdb
        # pdb.set_trace()
        


        fontsize = 24
        fig_list = []  # Create a list to store the figures for each vname
        for vname in vnames:
            # read reanalysis data
            real_data = np.load(
                f'./data/input/era5/{fc_timestamp}/{vname}.npy', 
                )

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
    os.makedirs(f'./data/demos/',exist_ok=True)
    save_path = f'./data/demos/surface_forecast_vs_real.gif'
    imageio.mimsave(save_path, images, fps=2)
    # compress_gif(save_path,save_path)

if __name__ == "__main__":
    plot_pressure_demo_gif(initial_timestamp='2023-06-01T00:00:00', 
                  steps=40, 
                  plot_variable={'z':[850,500],
                                 'q':[850,500],
                                 'u':[850,500],
                                 'v':[850,500],
                                 't':[850,500],
                                 }
                    )
    plot_surface_demo_gif(initial_timestamp='2023-06-01T00:00:00', 
                  steps=40, 
                  plot_variable=['sp','v10','v100', 't2m','tp6h', 'msl']
                                )
