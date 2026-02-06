import h5py
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

"""
This is to combine the pointclouds from different rections (tracks 1-2 from one and 3-5 from another)
"""
def combine_h5(run_num, counter_idx):
    file_3plus = h5py.File(f"//Volumes/researchEXT/spyral_eng/my_sim/output/kinematics/detector/resonan_more data/run_000{run_num}.h5", "r") if run_num < 10 else h5py.File(f"/Volumes/researchEXT/spyral_eng/my_sim/output/kinematics/detector/resonan_more data/run_00{run_num}.h5", "r")
    groupn_3plus = list(file_3plus.keys())[0]
    group_cr_3plus = file_3plus[groupn_3plus]
    attributes_3plus = dict(group_cr_3plus.attrs)
    min_event_3plus = attributes_3plus["min_event"]
    max_event_3plus = attributes_3plus["max_event"]

    file_123 = h5py.File(f"/Volumes/researchEXT/spyral_eng/my_sim/output/kinematics/detector/gs_more data/run_000{run_num}.h5", "r") if run_num < 10 else h5py.File(f"/Volumes/researchEXT/spyral_eng/my_sim/output/kinematics/detector/gs_more data/run_00{run_num}.h5", "r")
    groupn_123 = list(file_123.keys())[0]
    group_cr_123 = file_123[groupn_123]
    attributes_123 = dict(group_cr_123.attrs)
    min_event_123 = attributes_123["min_event"]
    max_event_123 = attributes_123["max_event"]

    output_path = f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/run_000{run_num}.h5" if run_num < 10 else f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/run_00{run_num}.h5"
    file_out = h5py.File(output_path, "w")
    group_out = file_out.create_group("cloud")
    group_out.attrs["min_event"] = counter_idx

    output_label_path = f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/run_000{run_num}_labels.h5" if run_num < 10 else f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/run_00{run_num}_labels.h5"
    file_out_label = h5py.File(output_label_path, "w")
    group_label = file_out_label.create_group("label")
    group_label.attrs["min_event"] = counter_idx

    total_min = min(min_event_3plus, min_event_123)
    total_max = max(max_event_3plus, max_event_123)

    for i in tqdm.tqdm(range(total_min, total_min + 1)):
        event = f"cloud_{i}"
        label_key = f"labels_{i}"

        if event in group_cr_3plus and label_key in group_cr_3plus:
            labels = np.unique(group_cr_3plus[label_key])
            print(labels)
            size = len(labels)
            if size >= 2 and size < 6 and size!=0: #otherwise stuff with labels above passes through
                new_key = f"event_{counter_idx}"
                group_out.create_dataset(new_key, data=group_cr_3plus[event][:])
                group_label.create_dataset(new_key, data=group_cr_3plus[label_key][:])
                group_out[new_key].attrs["source"] = "3plus"
                counter_idx += 1

        if event in group_cr_123 and label_key in group_cr_123:
            labels = np.unique(group_cr_123[label_key])
            print(labels)
            size = len(labels)
            if size <= 2 and size!=0:
                new_key = f"event_{counter_idx}"
                group_out.create_dataset(new_key, data=group_cr_123[event][:])
                group_label.create_dataset(new_key, data=group_cr_123[label_key][:])
                group_out[new_key].attrs["source"] = "123"
                counter_idx += 1

    group_out.attrs["max_event"] = counter_idx
    group_label.attrs["max_event"] = counter_idx

    file_3plus.close()
    file_123.close()
    file_out.close()
    file_out_label.close()

    return counter_idx + 1

def labels_sequential(run_num):
    file_labels = h5py.File(f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/run_000{run_num}_labels.h5", "r") if run_num < 10 else h5py.File(f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/run_00{run_num}_labels.h5", "r")
   
    file_labels.close()
if __name__ == "__main__":
    #run_range = [3,4,5]
    counter = 0
    for run in range(0,1):
        print(f"\n--- Starting run {run} ---")
        counter = combine_h5(run, counter)
        # convert(run)
    print(f"Final event count: {counter}")