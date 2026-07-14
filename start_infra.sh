#!/bin/bash
# start_infra.sh — Run on 192.168.1.26
# Starts etcd for Dynamo service discovery.
# Must be running before decode or prefill workers start.

set -e
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

echo "Starting etcd and NATS..."
sudo docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d etcd nats
echo "Waiting for etcd to be ready..."
until sudo docker exec etcd etcdctl endpoint health 2>/dev/null; do
  sleep 2
done
echo "Waiting for NATS to be ready..."
until sudo docker exec nats nc -z localhost 4222 2>/dev/null; do
  sleep 2
done
echo "==> etcd and NATS are ready"
