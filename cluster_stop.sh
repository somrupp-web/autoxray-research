#!/bin/bash
# cluster_stop.sh — gracefully stop all loops on all nodes

NODES=(
    "nvidia@10.137.203.188"
    "nvidia@10.137.203.189"
    "nvidia@10.137.203.190"
    "nvidia@10.137.203.191"
)

for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" \
        "tmux kill-session -t autoresearch 2>/dev/null; echo 'node-$i stopped'" &
done
wait
echo "All loops stopped."
