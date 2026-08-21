#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_HOME="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SCENE_DATA_HOME="${OBJ3DBB_DATA_HOME:-${SERVICE_HOME}/data}"
IMAGE="obj-centric-3dbb:1.0"
CONTAINER_NAME="proc-sim-obj3dbb-vis"
MODE="release"

usage() {
    echo "Usage: $0 [--mode <debug|release>]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            [[ $# -lt 2 ]] && usage
            MODE="$2"
            shift 2
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

if [[ "${MODE}" != "debug" && "${MODE}" != "release" ]]; then
    usage
fi

for path in src conf libs/boxer ckpts; do
    if [[ ! -d "${SERVICE_HOME}/${path}" ]]; then
        echo "Missing service path: ${SERVICE_HOME}/${path}"
        exit 1
    fi
done
if [[ ! -d "${SCENE_DATA_HOME}" ]]; then
    echo "Missing scene data path: ${SCENE_DATA_HOME}"
    echo "Set OBJ3DBB_DATA_HOME to the host directory mounted as /data"
    exit 1
fi
mkdir -p "${SERVICE_HOME}/logs"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    bash "${SERVICE_HOME}/scripts/stop.sh"
fi

DOCKER_ARGS=(
    --rm
    --name "${CONTAINER_NAME}"
    --gpus all
    --network host
    --workdir /workspace/boxer
    --env PYTHONUNBUFFERED=1
    --env PYTHONPATH=/workspace/boxer/src:/workspace/boxer/libs/boxer
    --volume "${SERVICE_HOME}/src:/workspace/boxer/src:ro"
    --volume "${SERVICE_HOME}/conf:/workspace/boxer/conf:ro"
    --volume "${SERVICE_HOME}/scripts:/workspace/boxer/scripts:ro"
    --volume "${SERVICE_HOME}/libs/boxer:/workspace/boxer/libs/boxer:ro"
    --volume "${SERVICE_HOME}/ckpts:/workspace/boxer/ckpts:ro"
    --volume "${SERVICE_HOME}/logs:/workspace/boxer/logs"
    --volume "${SCENE_DATA_HOME}:/data"
)

if [[ "${MODE}" == "debug" ]]; then
    echo "Starting ${CONTAINER_NAME} in an interactive shell"
    docker run -it "${DOCKER_ARGS[@]}" "${IMAGE}" /bin/bash
else
    echo "Starting ${CONTAINER_NAME}"
    docker run -d "${DOCKER_ARGS[@]}" "${IMAGE}" \
        /opt/conda/envs/xwk_boxer/bin/python \
        src/obj3dbb_vis_engine.py -c conf/config.yaml
fi
