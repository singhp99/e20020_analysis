import h5py
import numpy as np
import tqdm
from pathlib import Path


def filter_tracks(class_of_intrest, run_num):
    pc_path = f"/Volumes/researchEXT/O16/no_efield/PointcloudLegacy/run_0{run_num}.h5" #point clouds for experiment 
    file_exists = Path(pc_path) 
    pc_file = h5py.File(pc_path, "r")
    
    pc_ls = list(pc_file.keys())[0] #point cloud original
    pc_keys = pc_file[pc_ls]
    
    if file_exists.exists():
        valid_keys_path = f"/Volumes/researchEXT/O16/ml models/valid_keys/run{run_num}_valid_nonzero_keys.npy" #only valid keys for point cloud 
        valid_keys = np.load(valid_keys_path) #only valid point cloud keys

        pc_file = h5py.File(pc_path, "r")
    
        pc_ls = list(pc_file.keys())[0] #point cloud original
        pc_data = pc_file[pc_ls]
        
        min_event = np.int64(valid_keys[0].strip("cloud_")) #need to set min event as an attr
        max_event = np.int64(valid_keys[-1].strip("cloud_")) #need to set max event as an attr
        
        with h5py.File(f"run_00{run_num}.h5","w") as f:
            group = f.create_group("cloud")
            group.attrs["min_event"] = min_event
            group.attrs["max_event"] = max_event
            
            for key in valid_keys:
                group.create_dataset(key, data=pc_data[key][:])

                
        
    
    
def main():
    run_num = 104
    filter_tracks(5,104)
    
    
if __name__ == "__main__":
    main()
    