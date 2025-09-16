from spyral import (
    Pipeline,
    start_pipeline,
    GetParameters,
    ClusterParameters,
    EstimateParameters,
    PadParameters,
)

from e20009_phases.PointcloudLegacyPhase import PointcloudLegacyPhase
from e20009_phases.ClusterPhase import ClusterPhase
from e20009_phases.EstimationPhase import EstimationPhase
from e20009_phases.InterpSolverPhase import InterpSolverPhase
from e20009_phases.InterpLeastSqSolverPhase import InterpLeastSqSolverPhase
from e20009_phases.config import (
    ICParameters,
    DetectorParameters,
    SolverParameters,
)

from pathlib import Path
import multiprocessing

#########################################################################################################
# Set up workspace and trace paths
workspace_path = Path("/mnt/scratch/singhp19/single_dv_workspace")
trace_path = Path("/mnt/scratch/singhp19/O16_runs")

# Make directory to store beam events
if not workspace_path.exists():
    workspace_path.mkdir()
beam_events_folder = workspace_path / "beam_events"
if not beam_events_folder.exists():
    beam_events_folder.mkdir()

run_min = 53
run_max = 170
n_processes = 15

#########################################################################################################
# Define configuration
pad_params = PadParameters(
    pad_geometry_path=Path(
        "/mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/e20009_parameters/pad_geometry_legacy.csv"
    ),
    pad_time_path=Path(
        "/mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/e20009_parameters/pad_time_correction.csv"
    ),
    pad_electronics_path=Path(
        "/mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/e20009_parameters/pad_electronics_legacy.csv"
    ),
    pad_scale_path=Path(
        "/mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/e20009_parameters/pad_scale.csv"
    ),
)

get_params = GetParameters(
    baseline_window_scale=20.0,
    peak_separation=5.0,
    peak_prominence=20.0,
    peak_max_width=100.0,
    peak_threshold=100.0, #changed fom 30 
)

ic_params = ICParameters(
    baseline_window_scale=100.0,
    peak_separation=5.0,
    peak_prominence=30.0,
    peak_max_width=20.0,
    peak_threshold=300.0, #don't have the IC so doesn't matter
    low_accept=60,
    high_accept=411,
)

det_params = DetectorParameters(
    magnetic_field=3.0,
    electric_field=57260.0, #changing the efield
    detector_length=1000.0,
    beam_region_radius=20.0,
    drift_velocity_path=Path(
        "/mnt/home/singhp19/O16_driftvel_analysis/drift_vel_calc/all_drift_vel_with_sem.parquet"
    ),
    get_frequency=3.125,
    garfield_file_path=Path(
        "/mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/e20009_parameters/e20009_efield_correction.txt"
    ),
    do_garfield_correction=True, #SET FALSE
)

cluster_params = ClusterParameters(
    min_cloud_size=20,
    min_points=3,
    min_size_scale_factor=0.05,
    min_size_lower_cutoff=10,
    cluster_selection_epsilon=10.0,
    min_cluster_size_join=15,
    circle_overlap_ratio=0.25,
    outlier_scale_factor=0.05,
)

estimate_params = EstimateParameters(
    min_total_trajectory_points=20, smoothing_factor=100.0
)

# Protons
solver_params = SolverParameters(
    gas_data_path=Path(
        "./mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/solver_gas_16O.json"
    ),
    gain_match_factors_path=Path(
        "/mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/e20009_parameters/gain_match_factors.csv"
    ),
    particle_id_filename=Path("/mnt/home/singhp19/O16_driftvel_analysis/e20020_analysis/solver_particle_16O.json"),
    ic_min_val=300.0,
    ic_max_val=850.0,
    n_time_steps=1300,
    interp_ke_min=0.01,
    interp_ke_max=40.0,
    interp_ke_bins=800,
    interp_polar_min=0.1,
    interp_polar_max=179.9,
    interp_polar_bins=500,
)

# # Deuterons
# solver_params = SolverParameters(
#     gas_data_path=Path(
#         "C:\\Users\\zachs\\Desktop\\e20009_analysis\\e20009_analysis\\e20009_parameters\\e20009_target.json"
#     ),
#     gain_match_factors_path=Path(
#         "C:\\Users\\zachs\\Desktop\\e20009_analysis\\e20009_analysis\\e20009_parameters\\gain_match_factors.csv"
#     ),
#     particle_id_filename=Path("E:\\max_fix\\deuteron_id.json"),
#     ic_min_val=300.0,
#     ic_max_val=850.0,
#     n_time_steps=1300,
#     interp_ke_min=0.01,
#     interp_ke_max=80.0,
#     interp_ke_bins=1600,
#     interp_polar_min=0.1,
#     interp_polar_max=89.9,
#     interp_polar_bins=250,
# )

#########################################################################################################
# Construct pipeline
pipe = Pipeline(
    [
        PointcloudLegacyPhase(
            get_params,
            ic_params,
            det_params,
            pad_params,
        ),
        ClusterPhase(
            cluster_params,
            det_params,
        ),
        EstimationPhase(estimate_params, det_params),
        InterpSolverPhase(solver_params, det_params),
     ],
    [True, True, True, False],
    workspace_path,
    trace_path,
)


def main():
    start_pipeline(pipe, run_min, run_max, n_processes)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
