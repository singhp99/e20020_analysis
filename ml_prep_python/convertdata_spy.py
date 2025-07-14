import h5py
import numpy as np
import tqdm

def convert_n_combine(run_num):
    file = h5py.File(f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/run_000{run_num}.h5", "r")
    file_label = h5py.File(f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/run_000{run_num}_labels.h5", "r")

    gr_keys = list(file.keys())[0]
    keys = file[gr_keys]

    gr_labels = list(file_label.keys())[0]
    keys_labels = file_label[gr_labels]

    overlapping_keys = [e for e in keys_labels if e in keys]
    event_lengths = np.zeros(len(overlapping_keys), int)

    for i, e in enumerate(overlapping_keys):
        event_lengths[i] = len(keys[e])

    np.save(f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/processed_data/run0{run_num}_evtlen.npy", event_lengths)

    event_data = np.full((len(event_lengths), np.max(event_lengths) + 2, 4), np.nan)

    for i, e in range(1): #tqdm.tqdm(enumerate(overlapping_keys))
        print(event_lengths[i])
        for n in range(event_lengths[i]):
            event_data[i, n] = keys[e][n]
        label = int(keys_labels[e][()])
        event_data[i, -2] = [label] * 4
        event_data[i, -1] = [i] * 4

    np.save(f"/Volumes/researchEXT/spyral_eng/engine_ml_prep/processed_data/run0{run_num}_data.npy", event_data)

def main():
    for run in [0,1,2]:  # Update this list with the runs you want
        print(f"\n--- Starting run {run} ---")
        convert_n_combine(run)


if __name__ == "__main__":
    main()