import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def handlanel_stat():
    run_num = 104
    noise = 1
    data = np.load(f"/Volumes/researchEXT/O16/ml models/data_exp_pred/run{run_num}_16O_size800_test_features.npy")
    predicted_labels = np.load(f"/Users/pranjalsingh/Desktop/research_space_spyral/experiment_predicted/exp{run_num}_pred_w{noise}.npy")
    view_class = 4 #number of tracks - 1
    handlab_stat = []
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print(f"The class distribution for {noise}% data: {unique,counts}")
    
    for i in range(len(data)):
        if predicted_labels[i] == view_class:
            x = data[i,:,0]
            y = data[i,:,1]
            z = data[i,:,2]
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x, y, z, c="b", marker="o")
            ax.set_title(f"Index {i}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.tight_layout()
            plt.show()
            
            input_val = input("Hand label this event ior [q] for quit: ")
            
            if input_val == "q":
                unique_post, counts_post = np.unique(predicted_labels, return_counts=True)
                print(f"The class distribution for {noise}% data: {counts_post} from total of {len(handlab_stat)} events with predicted class {view_class}")
                break
            try:
                handlab_stat.append(int(input_val))
            except ValueError:
                print("Not a valid option. Try again")
                input_val = input("Hand label this event ior [q] for quit: ")
                if input_val == "q":
                    unique_post, counts_post = np.unique(predicted_labels, return_counts=True)
                    print(f"The class distribution for {noise}% data: {counts_post} from total of {len(handlab_stat)} events with predicted class {view_class}")
                    break
                handlab_stat.append(int(input_val))
            
            
    unique_post, counts_post = np.unique(handlab_stat, return_counts=True)
    print(f"The class distribution for {noise}% data: {unique_post,counts_post} from total of {len(handlab_stat)} events with predicted class {view_class}")
        
            
if __name__ == "__main__":
    handlanel_stat() 