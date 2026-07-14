#!/bin/bash
# start_infra.sh — Run on 192.168.1.24
# Starts etcd for Dynamo service discovery.
# Must be running before decode or prefill workers start.

set -e
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

echo "Starting etcd..."
sudo docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d etcd

echo "Waiting for etcd to be ready..."
until sudo docker exec etcd etcdctl endpoint health 2>/dev/null; do
  sleep 2
done

echo "==> etcd is ready"
