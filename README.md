# recon360 pycolmap wheels

Built by `scripts/build_patched_colmap.sh` in the recon360 repository from branch
`recon360-4.2` of this fork (COLMAP 4.2.1 + the fisheye theta > 90 deg patch, CUDA
sm_86 + sm_120, Ceres master with cuDSS). Stored in 90 MB parts because of GitHub's
100 MB file limit; `scripts/install_pycolmap.sh` downloads, reassembles and checks
them against SHA256SUMS. Not an official COLMAP release.
