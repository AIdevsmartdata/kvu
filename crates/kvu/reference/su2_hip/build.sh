#!/usr/bin/env bash
# Compile le générateur SU(2) HIP. Aucun HSA_OVERRIDE (gfx1151 natif sous ROCm 7.1).
set -eu
cd "$(dirname "$0")"
hipcc --version > logs/build_hipcc_version.txt 2>&1
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -o bin/su2_hip src/su2_hip.cpp > logs/build.log 2>&1
sha256sum src/su2_hip.cpp bin/su2_hip > bin/SHA256SUMS
echo "build OK"
