import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import h5py

def handlanel_stat():
    run_num = 30
    noise = 0
    # data = np.load(f"/Volumes/researchEXT/O16/ml models/data_exp_pred/run{run_num}_16O_size800_test_features.npy")
    data = np.load(f"/Volumes/researchEXT/24Mg/experiment_24Mg_test/run{run_num}_24Mg_size800_test_features.npy")
    # predicted_labels = np.load(f"/Users/pranjalsingh/Desktop/research_space_spyral/experiment_predicted/exp{run_num}_pred_w{noise}.npy")
    predicted_labels = np.load(f"/Volumes/researchEXT/24Mg/experiment_24Mg_test/predicted_labels_ml/exp{run_num}_pred_w{noise}.npy")
    valid_keys_path = Path(f"/Volumes/researchEXT/24Mg/experiment_24Mg_test/non-zero keys/run{run_num}_valid_nonzero_keys.npy")
    
    file_pc = h5py.File(f"/Volumes/researchEXT/24Mg/mg24_spyral/Pointcloud/run_00{run_num}.h5", "r")
    groupn_pc = list(file_pc.keys())[0]
    group_pc = file_pc[groupn_pc]
    
    valid_keys = np.load(valid_keys_path)
    view_class = 5 #number of tracks - 1
    handlab_stat = []
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print(f"The class distribution for {noise}% data: {unique,counts}")
    
    for i in range(len(valid_keys)):
        if predicted_labels[i] == view_class:
            x = data[i,:,0]
            y = data[i,:,1]
            z = data[i,:,2]
            
            event =  valid_keys[i] #"cloud_" +
            x_1 = group_pc[event][:, 0]
            y_2 = group_pc[event][:, 1]
            z_3 = group_pc[event][:, 2]
            
            fig = plt.figure(figsize=(12, 6))

            # First subplot
            ax1 = fig.add_subplot(121, projection="3d")
            ax1.scatter(x, y, z, c="b", marker="o", s=5)
            ax1.set_title(f"Original Data Index {i}")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")

            # Second subplot
            ax2 = fig.add_subplot(122, projection="3d")
            ax2.scatter(x_1, y_2, z_3, c="r", marker="^", s=5)
            ax2.set_title(f"Group Data Event {valid_keys[i]}")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")

            plt.suptitle(f"Event number: {event} ")
            plt.tight_layout()
            plt.show()
            
            input_val = input(f"{i} Hand label this event or [q] for quit: ")
            
            if input_val == "q":
                unique_post, counts_post = np.unique(predicted_labels, return_counts=True)
                print(f"The class distribution for {noise}% data: {counts_post} from total of {len(handlab_stat)} events with predicted class {view_class}")
            elif input_val == "n":
                continue
            try:
                handlab_stat.append(int(input_val))
            except ValueError:
                print("Not a valid option. Try again")
                input_val = input("Hand label this event or [q] for quit: ")
                if input_val == "q":
                    unique_post, counts_post = np.unique(predicted_labels, return_counts=True)
                    print(f"The class distribution for {noise}% data: {counts_post} from total of {len(handlab_stat)} events with predicted class {view_class}")
                    break
                handlab_stat.append(int(input_val))
            
            
    unique_post, counts_post = np.unique(handlab_stat, return_counts=True)
    
    print(f"The class distribution for {noise}% data: {unique,counts}")
    print(f"The class distribution for {noise}% data: {unique_post,counts_post} from total of {len(handlab_stat)} events with predicted class {view_class}")
        
            
if __name__ == "__main__":
    handlanel_stat() 