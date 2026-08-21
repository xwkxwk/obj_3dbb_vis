#!/bin/bash

set -euo pipefail

CONTAINER_NAME="proc-sim-obj3dbb-vis"

if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "Container '${CONTAINER_NAME}' does not exist"
    exit 0
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
    echo "Stopping ${CONTAINER_NAME} through the Engine stop lifecycle"
    if ! docker exec "${CONTAINER_NAME}" \
        /opt/conda/envs/xwk_boxer/bin/python -c \
        "import urllib.request; request = urllib.request.Request('http://127.0.0.1:8883/api/realtime/stop', data=b'{}', headers={'Content-Type': 'application/json'}, method='POST'); urllib.request.urlopen(request).read()"; then
        echo "The graceful stop request did not return normally"
    fi

    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]; then
        echo "Scene stopped; stopping service container ${CONTAINER_NAME}"
        docker stop --time 1 "${CONTAINER_NAME}" >/dev/null
    fi
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "${CONTAINER_NAME} stopped"
