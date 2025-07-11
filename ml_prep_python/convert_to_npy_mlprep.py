import h5py
import numpy as np
import tqdm

def combine_h5(run_num, counter_idx):
    file_3plus = h5py.File(f"/Volumes/researchEXT/spyral_eng/engine_spyral/PointcloudLegacy/run_000{run_num}.h5", "r")
    groupn_3plus = list(file_3plus.keys())[0]
    group_cr_3plus = file_3plus[groupn_3plus]
    attributes_3plus = dict(group_cr_3plus.attrs)
    min_event_3plus = attributes_3plus["min_event"]
    max_event_3plus = attributes_3plus["max_event"]

    file_12 = h5py.File(f"/Volumes/researchEXT/spyral_eng/my_sim/output/kinematics/detector/run_000{run_num}.h5", "r")
    groupn_12 = list(file_12.keys())[0]
    group_cr_12 = file_12[groupn_12]
    attributes_12 = dict(group_cr_12.attrs)
    min_event_12 = attributes_12["min_event"]
    max_event_12 = attributes_12["max_event"]

    output_path = f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/run_000{run_num}.h5"
    file_out = h5py.File(output_path, "w")
    group_out = file_out.create_group("cloud")
    group_out.attrs["min_event"] = counter_idx

    output_label_path = f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/run_000{run_num}_labels.h5"
    file_out_label = h5py.File(output_label_path, "w")
    group_label = file_out_label.create_group("label")
    group_label.attrs["min_event"] = counter_idx

    total_min = min(min_event_3plus, min_event_12)
    total_max = max(max_event_3plus, max_event_12)

    for i in tqdm.tqdm(range(total_min, total_max + 1)):
        event = f"cloud_{i}"
        label_key = f"labels_{i}"

        if event in group_cr_3plus and label_key in group_cr_3plus:
            labels = np.unique(group_cr_3plus[label_key])
            size = len(labels)
            if size >= 3:
                new_key = f"event_{counter_idx}"
                group_out.create_dataset(new_key, data=group_cr_3plus[event][:])
                group_label.create_dataset(new_key, data=size)
                group_out[new_key].attrs["source"] = "3plus"
                counter_idx += 1

        if event in group_cr_12 and label_key in group_cr_12:
            labels = np.unique(group_cr_12[label_key])
            size = len(labels)
            if size <= 2:
                new_key = f"event_{counter_idx}"
                group_out.create_dataset(new_key, data=group_cr_12[event][:])
                group_label.create_dataset(new_key, data=size)
                group_out[new_key].attrs["source"] = "12"
                counter_idx += 1

    group_out.attrs["max_event"] = counter_idx
    group_label.attrs["max_event"] = counter_idx

    file_3plus.close()
    file_12.close()
    file_out.close()
    file_out_label.close()

    return counter_idx + 1



if __name__ == "__main__":
    run_range = [0, 1, 2]
    counter = 0
    for run in run_range:
        counter = combine_h5(run, counter)
    print(f"Final event count: {counter}")