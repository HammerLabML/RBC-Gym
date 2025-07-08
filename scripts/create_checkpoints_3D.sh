#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the path to your Julia environment
export JULIA_PROJECT="${SCRIPT_DIR}/../.venv/julia_env"

# CHANGE PARAMETERS HERE
dir=${SCRIPT_DIR}/../data/checkpoints/3D/
ra=2500
pr=0.7
N=(32 32 16)
b=(1 2)
delta_t_snap=0.25
duration=200

# Now run your Julia script
julia ${SCRIPT_DIR}/../src/rbc_gym/sim/rbc_sim3D.jl --dir ${dir}/train --seed 42 --random_inits 20 --Ra ${ra} --N "${N[@]}" --b "${b[@]}" --delta_t_snap ${delta_t_snap} --duration ${duration} --use_cpu
julia ${SCRIPT_DIR}/../src/rbc_gym/sim/rbc_sim3D.jl --dir ${dir}/test --seed 62 --random_inits 10 --Ra ${ra} --N "${N[@]}" --b "${b[@]}" --delta_t_snap ${delta_t_snap} --duration ${duration} --use_cpu
julia ${SCRIPT_DIR}/../src/rbc_gym/sim/rbc_sim3D.jl --dir ${dir}/val --seed 72 --random_inits 10 --Ra ${ra} --N "${N[@]}" --b "${b[@]}" --delta_t_snap ${delta_t_snap} --duration ${duration} --use_cpu
