import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py
import random
import pandas as pd
import json
import os
from matplotlib.backends.backend_pdf import PdfPages
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

"""
To check if there are any nans in the pointcloud legacy phase
"""
def rmv_nans(name1):
    file_coord = h5py.File(name1, 'r')
    groupn_coord = list(file_coord.keys())[0]  # Get the first group name
    group_cr = file_coord[groupn_coord]  # Access the group
    
    counternan = 0
    for event in tqdm(group_cr, desc="Processing events"):
        data = group_cr[event][:]
        
        # Check for NaNs in the lowest (deepest) dimension
        if any(np.isnan(row).any() for plane in data for row in plane):
            counternan += 1
    print(counternan)
    file_coord.close()

"""
creating h5 file for estimator data
"""
def estimate_h5(estimate_df, run_num, est_path):
    file_est = pd.read_parquet(estimate_df, engine="pyarrow")
    grouped = file_est.groupby('event')
    
    group_sizes = grouped.size()
    h5_file_path = est_path
    
    with h5py.File(h5_file_path, 'w') as h5_file:
        estimator_group = h5_file.create_group('estimator')
        
        # Using tqdm with total number of events for a percentage-based progress bar
        for event, size in tqdm(group_sizes.items(), total=len(group_sizes), desc="Processing events"):
            key = 'cloud_' + str(event)
            estimator_group.create_dataset(key, data=size)

    print("HDF5 file created successfully.")

"""
Visualizing the point cloud with cluster (from estimation)
"""
def vis_cluspc(name1,est_path):
    file_coord = h5py.File(name1, 'r')
    groupn_coord = list(file_coord.keys())[0]  # Get the first group name
    group_cr = file_coord[groupn_coord]  # Access the group

    file_est = h5py.File(est_path,'r')
    groupn_est = list(file_est.keys())[0]
    group_est = file_est[groupn_est]

    for i,event in enumerate(group_est):
        x = group_cr[event][:, 0]
        y = group_cr[event][:, 1]
        z = group_cr[event][:, 2]
        charge = group_cr[event][:, 3]

        est_data = group_est[event][()]
        if (i <=1): #int(est_data) ==1 and 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x/250, y/250, (z-500)/500, marker='o', s=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_ylim(-1,1)
            ax.set_xlim(-1,1)
            ax.set_zlim(-1,1)
            ax.set_title(f'Event {event}')

            #ax.text2D(0.05, 0.95, f'Estimator Data: {est_data}', transform=ax.transAxes, color='red')

            plt.show()

    file_coord.close()
    file_est.close()

"""
Makes a histogram of how many events are in each class
"""
def num_points_eachevent(name1,est_path):
    file_coord = h5py.File(name1, 'r')
    groupn_coord = list(file_coord.keys())[0]  # Get the first group name
    group_cr = file_coord[groupn_coord]  # Access the group

    file_est = h5py.File(est_path, 'r')
    groupn_est = list(file_est.keys())[0]
    group_est = file_est[groupn_est]

    # Initialize lists for each estimator value
    event_lengths_1 = []
    event_lengths_2 = []
    event_lengths_3 = []
    event_lengths_4 = []
    event_lengths_5 = []

    # Iterate over each event
    for event in group_est:
        # Get the estimator data
        est_data = group_est[event][()]
        event_length = len(group_cr[event])

        if int(est_data) == 3:
            event_lengths_3.append(event_length)
        elif int(est_data) == 1:
            if len(event_lengths_1) == len(event_lengths_3):
                continue
            else:
                event_lengths_1.append(event_length)
        elif int(est_data) == 2:
            if len(event_lengths_1) == len(event_lengths_3):
                continue
            else:
                event_lengths_2.append(event_length)
       
        elif int(est_data) == 4:
            event_lengths_4.append(event_length)
        elif int(est_data) == 5:
            event_lengths_5.append(event_length)

    # Plot histograms for each estimator value in different colors
    plt.figure()
    plt.hist(event_lengths_1, bins=20, alpha=0.6, label='1 Track', color='blue', edgecolor='black')
    plt.hist(event_lengths_2, bins=20, alpha=0.6, label='2 Track', color='green', edgecolor='black')
    plt.hist(event_lengths_3, bins=20, alpha=0.6, label='3 Track', color='red', edgecolor='black')
    plt.hist(event_lengths_4, bins=20, alpha=0.6, label='4 Track', color='orange', edgecolor='black')
    plt.hist(event_lengths_5, bins=20, alpha=0.6, label='5 Track', color='purple', edgecolor='black')

    plt.xlabel('Number of Points per Event')
    plt.ylabel('Frequency')
    plt.title('Histogram of Event Lengths by Estimator')
    plt.xlim(0,2000)
    plt.legend()
    plt.show()

    file_coord.close()
    file_est.close()




"""
Counting how many events are in each class
"""
def count_class(estimate_df):


    file_est = pd.read_parquet(estimate_df, engine="pyarrow")
    grouped = file_est.groupby('event')
    group_sizes = grouped.size()

    class_count = [0,0,0,0,0]

    for event, size in group_sizes.items():

        if int(size) == 1:
            class_count[0]+=1
        
        elif int(size) == 2:
            class_count[1]+=1

        elif int(size) == 3:
            class_count[2]+=1
            
        elif int(size) == 4:
            class_count[3]+=1

        elif int(size) == 5:
            class_count[4]+=1

    
    print("Number of events in each class:",class_count)

    return class_count


"""
Balancing classes
"""
def balancing_classes(name1,est_path,bal_mltrain):
    file_coord = h5py.File(name1, 'r')
    groupn_coord = list(file_coord.keys())[0]  # Get the first group name
    group_cr = file_coord[groupn_coord]  # Access the group

    file_est = h5py.File(est_path,'r')
    groupn_est = list(file_est.keys())[0]
    group_est = file_est[groupn_est]

    class_count3 = count_class(name1,est_path,bal_mltrain)[2]
    class_bl = [0,0,0,0,0]

    

    # for key in group_est:
    #         event = group_est[key]
    #         nclus = event.attrs["nclusters"]
    #         if int(nclus) == 3:
    #             class_count3+=1
    
    
    with h5py.File(bal_mltrain, 'w') as h5_file:
        ml_group = h5_file.create_group('mltrain')
        for key in group_est:
            boolean = False
            event = group_est[key]
            nclus = group_est[key][()]

            if int(nclus) == 1:
                if class_bl[0] < class_count3:
                    boolean=True
                    class_bl[0] += 1
                elif class_bl[0] <= class_count3:
                    boolean=False
                    continue
                

            elif int(nclus) == 2:
                if class_bl[1] < class_count3:
                    boolean=True 
                    class_bl[1] += 1
                elif class_bl[1]>=class_count3:
                    boolean=False
                    continue

            elif int(nclus) == 3:
                if class_bl[2] <= class_count3:
                    boolean=True
                    class_bl[2] += 1
                elif class_bl[2]>class_count3:
                    boolean=False
                    continue

            elif int(nclus) == 4:
                class_bl[3] += 1 
                boolean = True

            elif int(nclus) == 5:
                class_bl[4] += 1
                boolean = True

            if boolean==True:
                x = group_cr[key][:, 0]
                y = group_cr[key][:, 1]
                z = group_cr[key][:, 2]
                charge = group_cr[key][:, 3]

                data = np.vstack((x, y, z, charge)).T

                ml_group.create_dataset(str(key),data=data)


    print(class_bl)

"""
counting classes post
"""
def cntcl_post(est_path,bal_mltrain):
    file_bl = h5py.File(bal_mltrain, 'r')
    groupn_bl = list(file_bl.keys())[0]  # Get the first group name
    group_bl = file_bl[groupn_bl]  # Access the group

    file_est = h5py.File(est_path,'r')
    groupn_est = list(file_est.keys())[0]
    group_est = file_est[groupn_est]

    count_bl = [0,0,0,0,0]
    for event in group_bl:
        tr_num = group_est[event][()]
        if tr_num == 1:
            count_bl[0]+=1
        elif tr_num == 2:
            count_bl[1]+=1
        elif tr_num == 3:
            count_bl[2]+=1
        elif tr_num == 4:
            count_bl[3]+=1
        elif tr_num == 5:
            count_bl[4]+=1
    print(count_bl)


"""
To visviluaize a puls run -> can't run this through spyral like this 
"""
def viz_pulsrun(pulse_run):
    counter = 0
    file_coord = h5py.File(pulse_run, 'r')
    groupn_coord = list(file_coord.keys())[0]  # Get the first group name
    group_cr = file_coord[groupn_coord]  # Access the group
    for event in group_cr:
        # if counter == 1:
        #     break
        x = group_cr[event][:, 0]
        y = group_cr[event][:, 1]
        z = group_cr[event][:, 2]
        charge = group_cr[event][:, 3]

        # fig = plt.figure()
        fig = plt.figure(figsize=(8, 10))  # Portrait layout
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])  # Grid layout

            # Top plot: X vs Y
        ax1 = fig.add_subplot(gs[0, :])  # Spanning both columns
        scatter_xy = ax1.scatter(x, y, c=charge, cmap='plasma', marker='o', s=8)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('X vs Y')
        cbar_xy = fig.colorbar(scatter_xy, ax=ax1, orientation='vertical', pad=0.01)
        cbar_xy.set_label('Charge')

            # Bottom left plot: Z vs X
        ax2 = fig.add_subplot(gs[1, 0])
        scatter_zx = ax2.scatter(z, x, c=charge, cmap='plasma', marker='o', s=8)
        ax2.set_xlabel('Z')
        ax2.set_ylabel('X')
        ax2.set_title('Z vs X')
        cbar_zx = fig.colorbar(scatter_zx, ax=ax2, orientation='vertical', pad=0.01)
        cbar_zx.set_label('Charge')

            # Bottom right plot: Z vs Y
        ax3 = fig.add_subplot(gs[1, 1])
        scatter_zy = ax3.scatter(z, y, c=charge, cmap='plasma', marker='o', s=8)
        ax3.set_xlabel('Z')
        ax3.set_ylabel('Y')
        ax3.set_title('Z vs Y')
        cbar_zy = fig.colorbar(scatter_zy, ax=ax3, orientation='vertical', pad=0.01)
        cbar_zy.set_label('Charge')

        plt.tight_layout()
        plt.show()

        
"""
for committee meeting plots
"""
def viz_niceplot(name1, est_path):
    file_coord = h5py.File(name1, 'r')
    groupn_coord = list(file_coord.keys())[0]  # Get the first group name
    group_cr = file_coord[groupn_coord]  # Access the group

    file_est = h5py.File(est_path, 'r')
    groupn_est = list(file_est.keys())[0]
    group_est = file_est[groupn_est]

    for event in group_est:
        x = group_cr[event][:, 0]
        y = group_cr[event][:, 1]
        z = group_cr[event][:, 2]
        charge = group_cr[event][:, 3]

        est_data = group_est[event][()]
        if int(est_data) == 5:
            fig = plt.figure(figsize=(8, 10))  # Portrait layout
            gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])  # Grid layout

            # Top plot: X vs Y
            ax1 = fig.add_subplot(gs[0, :])  # Spanning both columns
            scatter_xy = ax1.scatter(x, y, c=charge, cmap='plasma', marker='o', s=8)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title('X vs Y')
            cbar_xy = fig.colorbar(scatter_xy, ax=ax1, orientation='vertical', pad=0.01)
            cbar_xy.set_label('Charge')

            # Bottom left plot: Z vs X
            ax2 = fig.add_subplot(gs[1, 0])
            scatter_zx = ax2.scatter(z, x, c=charge, cmap='plasma', marker='o', s=8)
            ax2.set_xlabel('Z')
            ax2.set_ylabel('X')
            ax2.set_title('Z vs X')
            cbar_zx = fig.colorbar(scatter_zx, ax=ax2, orientation='vertical', pad=0.01)
            cbar_zx.set_label('Charge')

            # Bottom right plot: Z vs Y
            ax3 = fig.add_subplot(gs[1, 1])
            scatter_zy = ax3.scatter(z, y, c=charge, cmap='plasma', marker='o', s=8)
            ax3.set_xlabel('Z')
            ax3.set_ylabel('Y')
            ax3.set_title('Z vs Y')
            cbar_zy = fig.colorbar(scatter_zy, ax=ax3, orientation='vertical', pad=0.01)
            cbar_zy.set_label('Charge')

            plt.tight_layout()
            plt.show()

    file_coord.close()
    file_est.close()


"""
Reformatting the cluster flies from event_ to cloud_ 
"""
def reformat_pointclleg(name1, retries=5, delay=1):
    for attempt in range(retries):
        try:
            with h5py.File(name1, 'r+') as file_spy:
                spy_keys = list(file_spy.keys())[0]
                spy_gr = file_spy[spy_keys]
                keys_gr= list(spy_gr.keys()) 
                

                old_prefix = 'event_'
                new_prefix = 'cloud_'
                
                for key in keys_gr:
                    if key.startswith(old_prefix):
                        new_key = key.replace(old_prefix, new_prefix, 1)
                        spy_gr.move(key, new_key)
                        #print(f"Renamed {key} to {new_key}")
                    else:
                        print("doesn't start with cloud_")
                return  # Exit the function if successful
        except BlockingIOError:
            if attempt < retries - 1:
                print(f"File '{name1}' is locked. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise  # Raise the error if retries are exhausted



"""
Extracting labels from the cluster phase (w keys from the bal_mltrain estimator files)
"""

def extrt_clusph(name2,bal_mltrain_clus, bal_mltrain_cllabels):
    file_clus = h5py.File(name2, 'r') #r to read
    groupn_clus = list(file_clus.keys())[0]
    group_cl = file_clus[groupn_clus]

    file_blclus = h5py.File(bal_mltrain_clus, 'r') #r to read
    groupn_blclus = list(file_blclus.keys())[0]
    group_blcl = file_blclus[groupn_blclus]

    with h5py.File(bal_mltrain_cllabels, 'w') as h5_file:
        ml_group = h5_file.create_group('mllabels')
        for key in group_blcl:
            event = group_cl[key]
            nclus = event.attrs["nclusters"] 
            ml_group.create_dataset(key, data=nclus)

    print("HDF5 file created successfully.")

def engine_count_class(name1):
    file_coord = h5py.File(name1, 'r')
    groupn_coord = list(file_coord.keys())[0]  # Get the first group name
    group_cr = file_coord[groupn_coord]  # Access the group
    attributes = dict(group_cr.attrs)
    min_event = attributes["min_event"]
    max_event = attributes["max_event"]

    class_count = [0,0,0,0,0]

    for i in range(min_event,max_event+1):
        label_key = f"labels_{i}"
        event = f"cloud_{i}"
        
        if label_key in group_cr:
            # if len(group_cr[event][:]) < 20:
            #     print(len(group_cr[event][:]))
            #     continue

            tracks = np.unique(group_cr[label_key])
            size = len(tracks)

            if int(size) == 1:
                class_count[0]+=1
        
            elif int(size) == 2:
                class_count[1]+=1

            elif int(size) == 3:
                class_count[2]+=1
                
            elif int(size) == 4:
                class_count[3]+=1

            elif int(size) == 5:
                class_count[4]+=1

    
    print("Number of events in each class:",class_count)



def spyral_engine_viz(name1):
    file_coord = h5py.File(name1, 'r')
    groupn_coord = list(file_coord.keys())[0]  # Get the first group name
    group_cr = file_coord[groupn_coord]  # Access the group
    attributes = dict(group_cr.attrs)
    min_event = attributes["min_event"]
    max_event = attributes["max_event"]

    for i in range(min_event,min_event+100):
        event = f"cloud_{i}"
        if event in group_cr:
            if len(group_cr[event][:]) < 200:
                continue

            x = group_cr[event][:, 0]
            y = group_cr[event][:, 1]
            z = group_cr[event][:, 2]

            label_key = f"labels_{i}"
            label = np.unique(group_cr[label_key])
            size = len(label)

            if size != 3:
                continue

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x/250, y/250, (z-500)/500, marker='o', s=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_ylim(-1,1)
            ax.set_xlim(-1,1)
            ax.set_zlim(-1,1)
            ax.set_title(f'Event {event}')

            ax.text2D(0.05, 0.95, f'Truth Label (engine): {size}', transform=ax.transAxes, color='red')

            plt.show()

    file_coord.close()
            

def vertex_z_dist(estimate_df):
    df = pd.read_parquet(estimate_df, engine="pyarrow")
    filtered_vertex = df[(df["vertex_z"] < 0) | (df["vertex_z"] > 1000)]
    print(filtered_vertex["vertex_z"])
    # filtered_vertex["vertex_z"].hist(bins=100)
    # plt.title("Distribution of all vertex_z")
    # plt.xlabel("vertex_z [mm]")
    # plt.ylabel("Counts")
    # plt.xlim(0,8000)
    # plt.show()



def main():
    #for run_num in [271,274,275,277,278,279]:
    for run_num in [53]: #104,105,106,108,109,110,111,112,113,114,116
        print(run_num)
        name1 = f"/Users/mahesh/Desktop/academics/spyral_eng/engine_spyral/PointcloudLegacy/run_000{run_num}.h5"
        name2 = "/Users/mahesh/Desktop/academics/research/o16_analysis/Cluster/run_0"+str(run_num)+".h5"
        estimate_df = f"/Volumes/researchEXT/O16/O16_spyral_analysis/Estimation/run_00{run_num}.parquet"
        est_path = f"/Users/mahesh/Desktop/academics/spyral_eng/engine_ml_prep/run000{run_num}_est_spy.h5"
        bal_mltrain = f"/Users/mahesh/Desktop/academics/spring 2025/applied ml/ml_final_project/data_est/run000{run_num}_data_spy.h5"

        #pulse_run = "/Users/mahesh/Desktop/academics/research/o16_analysis/PointcloudLegacy/run_0173.h5"
        bal_mltrain_clus = f"/Users/mahesh/Desktop/academics/spring 2025/applied ml/ml_final_project/features_est/run0{run_num}_data_spy.h5"
        bal_mltrain_cllabels = f"/Volumes/researchEXT/O16/o16_ml_analysis/O16_balmltrain_clus/run0{run_num}_labels_spy.h5"
        #rmv_nans(name1)
        #estimate_h5(estimate_df,run_num,est_path)
        #vis_cluspc(name1,est_path)
        #count_class(estimate_df)
        #viz_pulsrun(pulse_run)
        #reformat_pointclleg(name1)
        ##balancing_classes(name1,est_path,bal_mltrain)
        #cntcl_post(est_path,bal_mltrain)
        #viz_niceplot(name1, est_path)
        #vis_cluspc(name1,est_path)
        #num_points_eachevent(name1,est_path)
        #extrt_clusph(name2, bal_mltrain_clus, bal_mltrain_cllabels)
        #engine_count_class(name1)
        #spyral_engine_viz(name1)
        vertex_z_dist(estimate_df)


if __name__=="__main__":    
    main()


