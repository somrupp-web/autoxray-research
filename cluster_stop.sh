#!/bin/bash
# cluster_stop.sh — stop all research loops on all nodes

NODES=(
    "nvidia@10.137.203.228"   # node-0
    "nvidia@10.137.203.184"   # node-1
    "nvidia@10.137.203.174"   # node-2
    "nvidia@10.137.203.177"   # node-3
)
SSH_PASS="nvidia"

for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    sshpass -p "$SSH_PASS" ssh \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=5 \
        "$node" "tmux kill-session -t autoresearch 2>/dev/null; echo 'node-$i stopped'" &
done
wait
echo "All loops stopped."
