import h5py
import numpy as np
import tqdm
from pathlib import Path


def filter_tracks(class_of_intrest, run_num):
    run_str = f"00{run_num}" if run_num < 100 else f"0{run_num}"
    pc_path = f"/Volumes/researchEXT/O16/no_efield/PointcloudLegacy/run_{run_str}.h5" #point clouds for experiment 
    file_exists = Path(pc_path) 

    predicted_path = Path(f"/Volumes/researchEXT/O16/ml models/O16_w1_noise_predicted_allruns/exp{run_num}_pred_w1.npy")

    valid_keys_path = Path(f"/Volumes/researchEXT/O16/ml models/valid_keys/run{run_num}_valid_nonzero_keys.npy") #only valid )keys for point cloud 

    if file_exists.exists() and predicted_path.exists() and valid_keys_path.exists:
        pc_file = h5py.File(pc_path, "r")
        pc_ls = list(pc_file.keys())[0] #point cloud original
        pc_data = pc_file[pc_ls]
        
        predicted_labels = np.load(predicted_path)
        
        valid_keys = np.load(valid_keys_path) #only valid point cloud keys
        
        min_event = np.int64(valid_keys[0].strip("cloud_")) #need to set min event as an attr
        max_event = np.int64(valid_keys[-1].strip("cloud_")) #need to set max event as an attr
        class_counter = 0
        with h5py.File(f"/Volumes/researchEXT/O16/no_efield/Pointcloud_sorted/run_{run_str}.h5","w") as f:
            new_group = f.create_group("cloud")
            new_group.attrs["min_event"] = min_event
            new_group.attrs["max_event"] = max_event
            
            for i, key in enumerate(tqdm.tqdm(valid_keys, desc=f"Filtering run {run_num}")):
                if predicted_labels[i] == class_of_intrest and key in pc_data:
                    new_group.create_dataset(key, data=pc_data[key][:])  
                    class_counter +=1
                
        print(f"Number of events for predicted label {class_of_intrest}: {class_counter}") 

    
def main():
    class_to_filter = 4
    for run_num in range(54,170):
        filter_tracks(class_to_filter,run_num)
    
    
if __name__ == "__main__":
    main()
    