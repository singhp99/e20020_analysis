import h5py
import numpy as np
import tqdm

def convert_n_combine(run_num):
    file = h5py.File(f"/Users/mahesh/Desktop/academics/spring 2025/applied ml/ml_final_project/data_est/run0{run_num}_data_spy.h5", "r")
    file_label = h5py.File(f"/Users/mahesh/Desktop/academics/spring 2025/applied ml/ml_final_project/est_h5 files/run0{run_num}_est_spy.h5", "r")

    gr_keys = list(file.keys())[0]
    keys = file[gr_keys]

    gr_labels = list(file_label.keys())[0]
    keys_labels = file_label[gr_labels]

    overlapping_keys = [e for e in keys_labels if e in keys]
    event_lengths = np.zeros(len(overlapping_keys), int)

    for i, e in enumerate(overlapping_keys):
        event_lengths[i] = len(keys[e])

    np.save(f"/Users/mahesh/Desktop/academics/spring 2025/applied ml/ml_final_project/processed_data/run0{run_num}_evtlen.npy", event_lengths)

    event_data = np.full((len(event_lengths), np.max(event_lengths) + 2, 4), np.nan)

    for i, e in tqdm.tqdm(enumerate(overlapping_keys)):
        for n in range(event_lengths[i]):
            event_data[i, n] = keys[e][n]
        label = int(keys_labels[e][()])
        event_data[i, -2] = [label] * 4
        event_data[i, -1] = [i] * 4

    np.save(f"/Users/mahesh/Desktop/academics/spring 2025/applied ml/ml_final_project/processed_data/run0{run_num}_data.npy", event_data)

def convert_n_combine_engine(run_num):
    file = h5py.File(f"/Users/mahesh/Desktop/academics/spring 2025/applied ml/ml_final_project/data_est/run0{run_num}_data_spy.h5", "r")

def main():
    for run in [104,105,106,108,109,110,111,112]:  # Update this list with the runs you want
        print(f"\n--- Starting run {run} ---")
        #convert_n_combine(run)


if __name__ == "__main__":
    main()