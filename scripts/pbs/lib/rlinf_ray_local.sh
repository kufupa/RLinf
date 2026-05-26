#!/bin/bash

setup_rlinf_local_ray() {
  local run_id="${PBS_JOBID:-local}"
  local run_safe="${run_id%%.*}"

  export RLINF_EPHEMERAL_BASE="${RLINF_EPHEMERAL_BASE:-${TMPDIR:-/tmp}/rlinf_ray_runs}"
  export RLINF_EPHEMERAL_RUN_ROOT="${RLINF_EPHEMERAL_RUN_ROOT:-${RLINF_EPHEMERAL_BASE}/${run_safe}}"
  export RLINF_RESULTS_DIR="${RLINF_RESULTS_DIR:-${RLINF_EPHEMERAL_RUN_ROOT}/results}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rlinf_ray_${run_safe}}"
  export TMPDIR="${RAY_TMPDIR}"
  unset RAY_ADDRESS || true

  case "${RAY_TMPDIR}" in
    /tmp/rlinf_ray_*)
      rm -rf "${RAY_TMPDIR}"
      ;;
    *)
      echo "error: refusing to delete unsafe RAY_TMPDIR=${RAY_TMPDIR}" >&2
      return 2
      ;;
  esac

  mkdir -p "${RLINF_EPHEMERAL_RUN_ROOT}/ray_tmp" "${RLINF_RESULTS_DIR}"
  ln -s "${RLINF_EPHEMERAL_RUN_ROOT}/ray_tmp" "${RAY_TMPDIR}"
}

archive_rlinf_ray_logs() {
  local run_id="${PBS_JOBID:-local}"
  local ephemeral_dst="${RLINF_EPHEMERAL_RUN_ROOT}/ray_logs"
  local project_dst="logs/ray/${run_id}"

  mkdir -p "${ephemeral_dst}" "${project_dst}"
  if [[ -L "${RAY_TMPDIR}/session_latest" && -d "${RAY_TMPDIR}/session_latest/logs" ]]; then
    cp -a "${RAY_TMPDIR}/session_latest/logs/." "${ephemeral_dst}/" || true
  elif [[ -d "${RAY_TMPDIR}/session_latest/logs" ]]; then
    cp -a "${RAY_TMPDIR}/session_latest/logs/." "${ephemeral_dst}/" || true
  fi
  cp -a "${ephemeral_dst}/." "${project_dst}/" || true
}
