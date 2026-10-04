# JetAuto Autonomous - Lane following + waypoint navigation

Standalone autonomous-driving package for JetAuto (Mecanum + Jetson Nano)
based on semantic segmentation (MobileNetV3 / external TensorRT).

**It does NOT use catkin_ws.** All scripts are launched directly with
`python2` (the system interpreter where ROS Melodic is installed).

## Platform

### Hardware

| Item | Value |
|---|---|
| Board | NVIDIA Jetson Nano Developer Kit |
| Module | NVIDIA Jetson Nano (16 GB eMMC) |
| SoC | Tegra210 (Porg) |
| CUDA Arch | 5.3 |
| L4T / JetPack | 32.7.4 / 4.6.4 |
| Hostname | `jetauto` |

### Software

| Item | Value |
|---|---|
| OS | Ubuntu 18.04 Bionic Beaver |
| Kernel | 4.9.337-tegra |
| Python (system / ROS) | 2.7 |
| Python (lane follower, conda) | 3.6.9 |
| ROS | Melodic |
| CUDA | 10.2.300 |
| cuDNN | 8.2.1.32 |
| TensorRT | 8.2.1.8 |
| OpenCV | 4.5.5 (CUDA: YES) |
| VPI / Vulkan | 1.2.3 / 1.2.70 |

## Architecture

```
                roscore + Hiwonder driver
                (started at boot)
                       │
    ┌------------------┼------------------┐
    ↓                  ↓                  ↓
/lane_mask_bev    /jetauto_controller   /odom
(publisher)       /cmd_vel              (publisher)
    ↑                  ↑
    │                  │ Twist
┌--------┐    ┌------------------------------------┐
│ lane_  │    │ lane_controller_node               │
│ follo- │--->│  └- lane_core (pure logic)         │
│ wer.py │    │ + waypoint_manager_node            │
│ (own   │    │ + n_orchestrator (sole driver)     │
│  env)  │    │ + serve_dashboard                  │
└--------┘    └------------------------------------┘
Python 3.6.9       Python 2.7
 conda env          system
```

The two Python "blocks" are completely separate at the process and
environment level; they communicate only via ROS topics.

## Package files

| File | Role |
|---|---|
| `scripts/lane_controller_node.py` | ROS node: rosparam/pub/sub wiring, handles TRACKING/SINGLE/STOP/DISABLED states |
| `scripts/lane_core.py` | Pure logic (no ROS): Hough, polynomial fit, steering with adaptive EMA - also importable from the offline tester |
| `scripts/waypoint_manager_node.py` | ROS node: Dijkstra + waypoint sequencing; publishes `nav_info` (plans, does not drive) |
| `scripts/n_orchestrator.py` | ROS node, self-contained sole publisher of `cmd_vel`: bundles the base `Orchestrator` (remap, drift fix, pure-pursuit, junction spin) plus lane/map disagreement blend, junctions, roundabout, drift corrections, fallback pure-pursuit, terminal EMERGENCY_STOP |
| `scripts/map_loader.py` | YAML loading + NetworkX graph, node classification |
| `scripts/perception_supervisor_node.py` | ROS node: starts/stops the Python 3 perception node from the dashboard (`run-models.sh` / `stop-models.sh`), reports `/perception/status` |
| `scripts/serve_dashboard.py` | Standalone mini HTTP server for the dashboard |
| `config/lane_params.yaml` | All parameters (speeds, Hough thresholds, BEV, orchestrator, etc.) |
| `web/dashboard.html` | UI: video feed + SVG map + controls + BEV calibration panel |
| `perception/bev_calibration_session.py` | Python 3: the dashboard-driven BEV calibration session (preview / apply / redo / abort) |
| `maps/map_clean-edited_smooth.yaml` | The track map (32 nodes, 36 directed edges) |
| `start_all.sh` | Starts the whole ROS stack (rosbridge, web_video, dashboard, lane_controller, waypoint_manager, orchestrator) |
| `stop_all.sh` | Stops everything + publishes a zero Twist |

## Prerequisites on the Jetson

Confirm them with these commands:

```bash
# 1) roscore must be running (it already is after Jetson boot)
pgrep -af rosmaster

# 2) Python 2 with all dependencies
python2 -c "import rospy, cv_bridge, yaml, networkx; print('OK')"

# 3) rosbridge_server and web_video_server installed
rospack find rosbridge_server
rospack find web_video_server
```

If `yaml` or `networkx` are missing for Python 2:

```bash
sudo apt install python-yaml python-networkx
```

## Installation

The package goes in any directory (recommended: `~/jetauto_autonomous`).

From the development PC, via scp:

```bash
cd ~/path/to/UniDrive
scp -r jetauto_autonomous jetauto@<JETSON_IP>:~/jetauto_autonomous
```

On the Jetson:

```bash
cd ~/jetauto_autonomous
chmod +x start_all.sh stop_all.sh scripts/*.py
```

## Start

In a terminal on the Jetson:

```bash
cd ~/jetauto_autonomous
./start_all.sh
```

What it launches:
1. Loads `config/lane_params.yaml` into `rosparam`
2. `rosbridge_websocket` (port 9090) - WebSocket bridge for the dashboard
3. `web_video_server` (port 8080) - MJPEG streaming of image topics
4. `serve_dashboard.py` (port 8000) - dashboard HTTP server
5. `lane_controller_node.py` - lateral control (proposes cmd_vel)
6. `waypoint_manager_node.py` - waypoint management (plans, publishes nav_info)
7. `n_orchestrator.py` - sole publisher of `/jetauto_controller/cmd_vel` (FSM, blend, roundabout, fallback)
8. `perception_supervisor_node.py` - lets the dashboard start/stop the perception node

Logs go to `/tmp/jetauto_autonomous_logs/`.
Process PIDs to `/tmp/jetauto_autonomous.pids`.

## Start the perception node (the models)

`start_all.sh` does **not** start it: it runs in a different Python
environment. Two ways:

**From the dashboard** (preferred) - press **▶ AVVIA MODELLI** in the
"Percezione" bar. This needs `perception/conda_env` set in
`config/lane_params.yaml` to the name of the conda env that has tensorrt +
rospy for Python 3; `run-models.sh` activates it itself.

**By hand**, in a separate terminal:

```bash
conda activate <env_name>
cd ~/jetauto_autonomous
./run-models.sh              # both models; ./stop-models.sh to stop
```

Verify with:

```bash
rostopic hz /lane_mask_bev
```

## BEV calibration from the dashboard

The bird's-eye-view warp is derived from four lane corners found in the
segmentation mask, cached in `perception/calibration.json`. Press
**📐 Calibra BEV** with the models running:

1. *"Avviare la calibrazione BEV?"* - **Sì** stops autonomous driving and
   captures the next mask.
2. Two previews appear: the corners picked on the mask, and the BEV they
   produce. Then *"Applicare la calibrazione?"*:
   - **Applica** - commit and save. Effective on the next frame, no engine
     reload.
   - **Rifai** - recompute from a fresh frame (aim at a stretch of road where
     both lane markings are clearly visible).
   - **Annulla** - discard. The previous calibration is kept untouched: the
     candidate is only staged until you apply it.

If no calibration exists when the models come up, the panel opens by itself.
A calibration that finds no lane pixels reports why and can simply be redone.

## Open the dashboard

From any PC on the same network (or from the Jetson itself via NoMachine):

```
http://<JETSON_IP>:8000/
```

You should see:
- a green "Connected" dot at the top right
- the video feed with the colored mask
- the SVG map with the track nodes
- Start/End controls, START, STOP

## Stop

```bash
cd ~/jetauto_autonomous
./stop_all.sh
```

## Quick tuning

Edit `config/lane_params.yaml` directly on the Jetson (it is plain YAML,
nothing needs recompiling). Restart with:

```bash
./stop_all.sh && ./start_all.sh
```

Parameters you will probably want to touch at the first test:

| Parameter | Effect | Default | Typical range |
|---|---|---|---|
| `linear_x_speed` | cruise speed (m/s) | 0.05 | 0.03 - 0.15 |
| `max_angular_z` | max steering rate (rad/s, drive=classic) | 0.80 | 0.4 - 1.2 |
| `max_steering_angle` | max mapped angle (degrees) | 48.0 | 30 - 60 |
| `single_line_offset` | px offset for the centre estimate with a single line | 0 | 0 - 60 |
| `hough_roi_top_frac` | top portion of the BEV ignored (far from the robot) | 0.30 | 0.0 - 0.6 |
| `hough_threshold` | minimum HoughLinesP votes | 50 | 30 - 80 |
| `hough_max_gap_px` | max gap to join Hough segments | 40 | 10 - 60 |
| `hough_min_length_px` | min validated line length | 20 | 10 - 40 |
| `center_y_ratio` | lane-centre measurement height [0=top,1=bottom] | 0.50 | 0.3 - 0.7 |
| `angle_smooth_alpha_base` | EMA base on the angle (lower = smoother) | 0.50 | 0.3 - 0.8 |
| `lane_width_px` | lane width in BEV px (static fallback) | 280 | 200 - 350 |
| `control_rate_hz` | control loop rate (Hz) | 20 | match the model FPS |

Measure the model FPS with:

```bash
rostopic hz /lane_mask_bev
```

And set `control_rate_hz` ≤ model_FPS + 5 (see the "Recommended tuning for
Jetson Nano 4GB" section below).

## Dynamic lane-width calibration

The controller continuously measures the distance between the left and
right line whenever both are visible and keeps an EMA estimate of the
lane width in BEV pixels. When the robot then sees a single line (e.g.
it is laterally offset and the other line leaves the frame), it uses the
dynamic estimate instead of the static `lane_width_px` to reconstruct
the lane centre. Without this mechanism, a `lane_width_px` wrong by even
15% with respect to the real track pushes the robot systematically off
centre.

**When it activates**: the first frame with two valid lines bootstraps
it. Every new measurement enters the EMA if it passes two sanity checks:

1. Absolute range `[lane_width_min_px, lane_width_max_px]` (always).
2. Relative band `lane_width_sanity_band` around the current value (only after bootstrap).

**Disabling**: `lane_width_dynamic_enable: false` -> back to the static
behavior (always uses `lane_width_px`).

**Verification from the debug image** (`/lane_debug/image`):
- Bottom left shows `W=...` (yellow): current EMA value in px.
- Below it: `Wm=...` last raw measurement, **green** if accepted into the EMA, **red** if rejected.
- On the magenta line (`center_y` height), two orange ticks at `±W/2` from the estimated lane centre.

**Tuning**:
- If `W` oscillates by ±20px frame to frame -> lower `lane_width_ema_alpha` (e.g. 0.05).
- If `Wm` is often red even on a good track -> widen `lane_width_sanity_band` (e.g. 0.35) or re-check the absolute bounds.
- If `W=--` permanently -> no measurement ever passed the sanity checks; check `lane_width_min_px` / `lane_width_max_px` (they are in **post-`bev_scale`** pixels).

**Scaling with bev_scale**: if you set `bev_scale: 2.0`, double
`lane_width_px`, `lane_width_min_px`, `lane_width_max_px`. The node
prints a startup warning if the bounds do not contain `lane_width_px`.

## Map fallback (inside the orchestrator)

The pure-pursuit fallback is no longer a separate node: it lives inside
the orchestrator as the `FALLBACK` FSM state. When the lane stays in
`HOLD`/`STOP` for `hold_ramp_ticks` consecutive ticks during navigation,
the orchestrator disables the lane controller and drives on the path
computed by `waypoint_manager_node` until the lane is stable again
(back to `NAVIGATING` after `recovery_ticks` OK ticks near a path node).
FALLBACK is bounded: a sustained cross-track excursion or a timeout trips
the terminal `EMERGENCY_STOP` instead of wandering off-road.

Parameters live in `lane_params.yaml` under `orchestrator:`
(`hold_ramp_ticks`, `recovery_ticks`, `lookahead_m`, `map_drive_speed`,
`map_kp`, `map_max_angular`).

## Recommended tuning for Jetson Nano 4GB

With MobileNetV3 + external TensorRT the model produces 20-30 FPS on the
Jetson Nano 4GB. Suggested values for `lane_params.yaml`:

```yaml
control_rate_hz: 15        # margin for ROS Melodic latency (Python 2)
bev_scale: 1.0             # raise to 2.0 only if the Jetson copes and it actually helps
hough_threshold: 50
hough_min_line_px: 20
hough_max_gap_px: 20       # on a 320×128 BEV a gap of 40 joins distant segments
hough_min_length_px: 20
hough_roi_top_frac: 0.30
center_y_ratio: 0.50
lane_fit_mode: "auto"
angle_smooth_alpha_base: 0.50
lane_width_ema_alpha: 0.10
```

Notes:
- `control_rate_hz` must be ≤ model FPS. With a real 25 FPS, 15 Hz leaves margin for ROS Melodic latency on Python 2.
- `hough_max_gap_px=40` (historic default) on a 320×128 BEV can join segments belonging to different lines: with dashed lines and noise, 20 is more conservative.
- If the Jetson CPU is saturated, set `publish_debug: false` or `debug_scale: 0.4`.

## Troubleshooting

**"roscore/rosmaster is NOT running"** -> the system is not ready yet. Wait
for the startup script to finish, or launch it manually: `roscore &`.

**"missing Python dependencies"** -> `sudo apt install python-yaml python-networkx`.

**Robot does not move after START** -> check:
- `rostopic hz /lane_mask_bev` must report a rate > 0
- `rostopic echo /lane_controller/state` must show `TRACKING_*`, not `STOP`
- `rostopic echo /jetauto_controller/cmd_vel` must show a non-zero Twist

**Robot stops mid-track with no visible lane** - with an active goal the
orchestrator switches to `FALLBACK` (pure-pursuit on the path) after
`hold_ramp_ticks` ticks of lost lane. If it still stops, verify that a
navigation is active (`/waypoint_manager/status == NAVIGATING`) and that
`/waypoint_manager/path` is not empty.

**Robot stopped with a red EMERGENCY STOP banner** - a failsafe tripped
(lane/map conflict or sustained off-reference excursion; the reason is in
the orchestrator log). It does not auto-recover by design: inspect the
log line `EMERGENCY STOP: <why>`, fix/recalibrate if needed, and issue a
new goal from the dashboard to resume.

**Robot oscillates** -> lower `Kp_lat` to 0.002 or raise `smooth_alpha` to 0.6.

**Dashboard "Disconnected"** -> verify port 9090 is reachable from the
browser: `curl http://<IP>:9090/` must return at least an HTTP response.

**Gray video feed** -> `tail -f /tmp/jetauto_autonomous_logs/web_video_server.log`
to look for errors. Check `rostopic hz /lane_debug/image`.