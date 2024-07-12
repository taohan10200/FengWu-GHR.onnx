# /bin/bash
initial_time=$1
source activate /home/admin/miniconda3/envs/ghr
python tools/shanxi/merge_pressure_surface.py  --initial_time=${initial_time}
python tools/shanxi/interp_stationv2.py  --initial_time=${initial_time}