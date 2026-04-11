#!/usr/bin/env bash
echo "=== Persistent Settings (always active) ==="

echo "  Swap:"
swapon --show 2>/dev/null || echo "    (none — correct)"

echo "  Kernel tuning:"
echo "    swappiness: $(cat /proc/sys/vm/swappiness)"
echo "    max_map_count: $(cat /proc/sys/vm/max_map_count)"
echo "    memlock: $(ulimit -l)"

echo ""
echo "=== On-Demand Settings (active only while Athena runs) ==="

echo "  CPU governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "  Transparent huge pages: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
echo "  NUMA balancing: $(cat /proc/sys/kernel/numa_balancing)"

echo "  GPU:"
nvidia-smi --query-gpu=clocks.gr,clocks.mem,persistence_mode,power.draw --format=csv 2>/dev/null

echo "  NVMe scheduler: $(cat /sys/block/nvme0n1/queue/scheduler 2>/dev/null)"

echo ""
echo "=== Resources ==="
free -h
