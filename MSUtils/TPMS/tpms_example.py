"""
Example script demonstrating the generation and optimization of a Triply Periodic Minimal Surface (TPMS) microstructure.
This script creates a TPMS microstructure of type 'iwp' in shell mode with specified resolution and dimensions.
It first generates an initial microstructure, computes volume fractions, and writes it to an HDF5 file.
Then, it optimizes the shell thickness to achieve a target volume fraction of 0.2 for phase 0,
regenerates the microstructure with optimized parameters, recomputes volume fractions, and writes the optimized
result to HDF5.
"""
from MSUtils.TPMS.tpms import TPMS
from MSUtils.general.MicrostructureImage import MicrostructureImage
from MSUtils.general.h52xdmf import write_xdmf

if __name__ == "__main__":
    N = 256, 256, 256
    L = 1.0, 1.0, 1.0
    tpms_type = "iwp"
    h5_filename = "data/tpms_opt.h5"
    unitcell_frequency = (1, 1, 1)
    invert = False

    tpms = TPMS(
        tpms_type=tpms_type,
        resolution=N,
        L=L,
        unitcell_frequency=unitcell_frequency,
        invert=invert,
        mode="shell",
        shell_thickness=0.1,
    )
    MS = MicrostructureImage(image=tpms.image, L=L)
    print(f"Volume fraction of phase 0: {MS.volume_fractions[0]:.4f}")
    print(f"Volume fraction of phase 1: {MS.volume_fractions[1]:.4f}")
    MS.write(
        h5_filename=h5_filename,
        dset_name="threshold_opt/" + tpms_type + "_thresh_0",
        order="zyx",
        compression_level=9,
    )

    # Optimize shell thickness to achieve target volume fraction for phase 1
    vf_target_phase_1 = 0.3
    threshold_opt, thickness_opt = tpms.find_threshold_for_volume_fraction(
        vf_target_phase_1,
    )
    print(
        f"New threshold and shell thickness for volume fraction {vf_target_phase_1}: {threshold_opt}, {thickness_opt}"
    )

    # Regenerate TPMS with optimized parameters
    tpms = TPMS(
        tpms_type=tpms_type,
        resolution=N,
        L=L,
        unitcell_frequency=unitcell_frequency,
        invert=invert,
        mode="shell",
        threshold=threshold_opt,
        shell_thickness=thickness_opt,
    )
    MS = MicrostructureImage(image=tpms.image, L=L)
    print(f"Volume fraction of phase 0: {MS.volume_fractions[0]:.4f}")
    print(f"Volume fraction of phase 1: {MS.volume_fractions[1]:.4f}")
    MS.write(
        h5_filename=h5_filename,
        dset_name="threshold_opt/" + tpms_type + "_thresh_opt",
        order="zyx",
        compression_level=9,
    )

    # Write XDMF for visualization
    write_xdmf(
        h5_filepath=h5_filename,
        xdmf_filepath="data/tpms_opt.xdmf",
        microstructure_length=L[::-1],
        time_series=False,
        verbose=True,
    )
