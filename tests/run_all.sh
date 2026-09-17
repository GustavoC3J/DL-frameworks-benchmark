#!/bin/bash
# Runs the equivalence suite in the three environments. Extra arguments go to pytest.
#   bash tests/run_all.sh [--gpu 2] [-k lstm] [-x] ...
# CONDA_HOME overrides where conda is installed.

cd "$(dirname "$0")/.."
source "${CONDA_HOME:-$HOME/miniconda3}/etc/profile.d/conda.sh"

# Stale results from another run would be compared against the current code
rm -rf tests/.artifacts

status=0

for pair in "bm_tf_env tensorflow" "bm_torch_env torch" "bm_jax_env jax"; do
    # Split without "set --", which would overwrite the script's own arguments
    env_name=${pair% *}
    backend=${pair#* }

    echo "==================== $env_name (KERAS_BACKEND=$backend) ===================="
    conda activate "$env_name"
    # -rfE: Short summary at the end with the failed tests and errors
    KERAS_BACKEND=$backend python -m pytest tests/ -p no:cacheprovider -rfE "${@}" || status=1
done

exit $status
