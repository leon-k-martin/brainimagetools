from os.path import join, abspath, dirname

PACKAGE_ROOT = dirname(abspath(__file__))
DATA_DIR = join(PACKAGE_ROOT, "data")

LEFT_SURFACE_INFL = join(
    DATA_DIR, "HCP1200_fs_LR_32k", "S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii"
)
RIGHT_SURFACE_INFL = join(
    DATA_DIR, "HCP1200_fs_LR_32k", "S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii"
)
