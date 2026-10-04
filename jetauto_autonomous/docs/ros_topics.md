# ROS Topic Map — JetAuto Autonomous Stack

All topics currently active in the autonomous driving stack.
**P** = publisher  **S** = subscriber  **L** = latched  **opt** = conditional/optional

---

## Topic table

| Topic | Type | LF | LC | WM | ORC | DB | Notes | On robot? |
|---|---|:---:|:---:|:---:|:---:|:---:|---|:---:|
| `/depth_cam/rgb/image_raw` | `sensor_msgs/Image` | S | | | | S | Raw camera feed (640×480) | 🚜 |
| `/odom` | `nav_msgs/Odometry` | | | S | S | S | Published by Hiwonder base driver | 🚜 |
| `/lane_mask_bev` | `sensor_msgs/Image` mono8 | P | S | | | | BEV-warped class mask 320×128. Primary LC input | ✔️ |
| `/lane_controller/cmd_vel` | `geometry_msgs/Twist` | | P | | S | | **Proposed only** — never sent to hardware. ORC forwards it | ✔️ |
| `/lane_controller/state` | `std_msgs/String` L | | P L | | S | S | OK / HOLD / STOP / DISABLED | ✔️ |
| `/lane_controller/enable` | `std_msgs/Bool` L | | S | | P L | P | ORC is the sole enable publisher (DB can also disable via FERMA button) | ✔️ |
| `/lane_controller/info` | `std_msgs/Float64MultiArray` | | P | | S | | Additive camera-frame lane geometry (8-element, see layout). ORC uses it for the roundabout guardrail only | ✔️ |
| `/lane_debug/image` | `sensor_msgs/Image` | | P opt | | | S opt | BEV + Hough overlay; streamed via web_video_server | ⁉️ |
| `/lane_follower/debug_image` | `sensor_msgs/Image` | P opt | | | | S opt | Coloured seg overlay + BEV mask (`--debug`); streamed via web_video_server | ⁉️ |
| `/object_detection/drive` | `std_msgs/String` | | | | S | | JSON `{"detections": [{class_name, score, box}]}` from `perception_node.py` (PN). ORC's traffic-light/STOP override consumes it | 🚜 |
| `/object_detection/video` | `sensor_msgs/Image` bgr8 | | | | | S opt | Annotated detection overlay from `perception_node.py` (PN); streamed via web_video_server | ⁉️ |
| `/remap_transform` | `std_msgs/Float64MultiArray` | | | S | S | P | `[theta, scale, tx, ty]` live coordinate frame update | ✔️ |
| `/waypoint_manager/goal` | `std_msgs/Int32MultiArray` | | | S | P | P | `[start_id, end_id]`; empty array = cancel. ORC publishes an empty array on EMERGENCY_STOP (and on opt-in REPLAN) | ✔️ |
| `/waypoint_manager/path` | `std_msgs/Int32MultiArray` L | | | P L | S | S | Current path as ordered node-ID list | ⁉️ |
| `/waypoint_manager/nav_info` | `std_msgs/Float64MultiArray` | | | P | S | | 8-element at 25 Hz — see layout below | ✔️ |
| `/waypoint_manager/status` | `std_msgs/String` L | | | P L | | S | State word: `IDLE` / `NAVIGATING` / `GOAL_REACHED` / `ERROR` | ✔️ |
| `/orchestrator/state` | `std_msgs/String` L | | | | P L | S | FSM state: `IDLE` / `NAVIGATING` / `JUNCTION` / `ROUNDABOUT` / `FALLBACK` / `EMERGENCY_STOP` / `DONE`, plus transient object-detection overrides `TRAFFIC_STOP` / `STOP_SIGN`. DB shows JUNCTION/FALLBACK/ROUNDABOUT badges + an EMERGENCY_STOP banner | ✔️ |
| `/jetauto_controller/cmd_vel` | `geometry_msgs/Twist` | | | | P | | **Sole hardware output — only ORC publishes here** | 🚜 |
| `/perception/cmd` | `std_msgs/String` | | | | | P | `start` / `stop` / `restart` → PS (perception supervisor) | ✔️ |
| `/perception/status` | `std_msgs/String` L | | | | | S | JSON from PS: `{state, pid, mode, calibrated, log, message}` at 1 Hz. `calibrated` drives the first-boot calibration prompt | ✔️ |
| `/bev_calibration/cmd` | `std_msgs/String` | S | | | | P | `start` / `redo` / `apply` / `abort` → PN | ✔️ |
| `/bev_calibration/status` | `std_msgs/String` L | P L | | | | S | JSON: `{state, reason, calibrated, src_points, staged_points, angle_deg, calibration_file}` | ✔️ |
| `/bev_calibration/preview_points` | `sensor_msgs/CompressedImage` L | P L | | | | S | JPEG: colorized mask + the four picked corners | ✔️ |
| `/bev_calibration/preview_warp` | `sensor_msgs/CompressedImage` L | P L | | | | S | JPEG: the BEV those corners produce | ✔️ |

**Column key:**
- **LF** — `perception_node.py` (Python 3, conda env; `jetauto_autonomous/perception/`). The merged perception node: YOLO11/TensorRT object detection + lane segmentation. Launched via `./run-models.sh`. Publishes both `/lane_mask_bev` and `/object_detection/*`. (The column is labelled "LF" for historical continuity — the former `lane_follower.py` occupied it; that script is now superseded.)
- **LC** — `lane_controller_node.py` (Python 2.7)
- **WM** — `waypoint_manager_node.py` (Python 2.7)
- **ORC** — `n_orchestrator.py` (Python 2.7; run exactly one — same node name). The sole node that drives the robot.
- **DB** — `dashboard.html` via rosbridge WebSocket
- **PS** — `perception_supervisor_node.py` (Python 2.7). Starts/stops the perception node (LF) via `run-models.sh` / `stop-models.sh` on dashboard command and reports whether it is alive. It has no column of its own: it only owns `/perception/*`.

**Topics removed vs. previous architecture:**
- `/map_follower/active` — was used by old `map_follower_node.py`; fallback logic now internal to ORC
- `/map_follower/state` — same; removed entirely
- Multiple publishers on `/jetauto_controller/cmd_vel` — now ORC only

---

## `/waypoint_manager/nav_info` layout

`std_msgs/Float64MultiArray`, published at 25 Hz while navigation is active.

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `node_id` | int (float) | Current target node ID; -1 if inactive |
| 1 | `next_id` | int (float) | Next node ID in path; -1 if last segment |
| 2 | `dist_to_current` | float (m) | Distance from robot to current target node |
| 3 | `is_junction` | 0 or 1 | 1 if current node is a junction (degree > 2) |
| 4 | `heading_to_next` | float (rad) | MAP-frame bearing from robot to next node |
| 5 | `path_idx` | int (float) | Current index in the path array |
| 6 | `path_len` | int (float) | Total nodes in current path |
| 7 | `is_active` | 0 or 1 | 1 while navigating, 0 when IDLE/DONE/ERROR |

---

## `/lane_controller/info` layout

`std_msgs/Float64MultiArray`, additive (the `cmd_vel`/`state` path is unchanged). Consumed by
the orchestrator's roundabout guardrail only. Offsets normalized to W/2 (|.|~0 = line at robot
centre, about to be crossed).

| Index | Name | Description |
|---|---|---|
| 0 | `state_code` | 0 STOP, 1 HOLD, 2 TRACKING_CC, 3 SINGLE_L, 4 SINGLE_R, 5 DISABLED |
| 1 | `heading_rad` | real lane direction vs robot forward (+ve bends right) |
| 2 | `left_valid` | 0/1 |
| 3 | `right_valid` | 0/1 |
| 4 | `left_offset` | normalized lateral offset of the left line |
| 5 | `right_offset` | normalized lateral offset of the right line |
| 6 | `center_offset` | normalized lane-centre offset |
| 7 | `lane_width` | normalized lane width |

---

## External services

| Service | Port | Purpose |
|---|---|---|
| `rosbridge_websocket` | 9090 | WebSocket bridge for dashboard JS |
| `web_video_server` | 8080 | MJPEG stream for image topics |
| `serve_dashboard.py` | 8000 | Serves `dashboard.html` and map YAML |

---

## BEV calibration session

The BEV warp is (re)calibrated at runtime from the dashboard. A candidate is
computed in the perception node and previewed; the live warp and
`perception/calibration.json` change only on `apply`, so `abort` is safe by
construction.

```
DB --"start"--> PN     capture the next segmentation mask, compute candidate
PN --"PREVIEW"-> DB    + two preview JPEGs (latched)
DB --"apply"---> PN    commit + save; effective on the next frame, no engine reload
DB --"redo"----> PN    recompute from a fresh mask
DB --"abort"---> PN    drop the candidate; live warp untouched
```

`/bev_calibration/status.state` ∈ `IDLE · CAPTURING · PREVIEW · APPLIED · FAILED · UNAVAILABLE`
(`UNAVAILABLE` when the perception node runs `--mode detection`).
`/perception/status.state` ∈ `STOPPED · STARTING · RUNNING · STOPPING · ERROR`.

---

## Data-flow diagram

```
JetAuto base
  /depth_cam/rgb/image_raw ──────────────────────► perception_node.py (LF)
  /odom ───────────────────────────┬─────────────► waypoint_manager (WM)
                                   └─────────────► orchestrator (ORC)

LF  (Python 3, detection + segmentation; ./run-models.sh)
  ├── /lane_mask_bev  ────────────────────────────► lane_controller (LC)
  ├── /lane_follower/debug_image  ────────────────► web_video_server → dashboard
  ├── /object_detection/drive  ───────────────────► orchestrator (ORC)  (traffic-light/STOP override)
  └── /object_detection/video  ───────────────────► web_video_server → dashboard

LC  (Python 2.7)
  ├── /lane_controller/cmd_vel  ──────────────────► orchestrator (ORC)
  ├── /lane_controller/state  ────────────────────► ORC, dashboard
  ├── /lane_controller/info  ─────────────────────► ORC  (roundabout guardrail)
  └── /lane_debug/image  ──────────────────────────► web_video_server → dashboard

WM  (Python 2.7)
  ├── /waypoint_manager/nav_info  ────────────────► ORC  (25 Hz)
  ├── /waypoint_manager/path  ────────────────────► ORC, dashboard
  └── /waypoint_manager/status  ──────────────────► dashboard

ORC  (n_orchestrator.py)  ← THE ONLY NODE THAT DRIVES THE ROBOT
  ├── /lane_controller/enable  ────────────────────► LC
  ├── /orchestrator/state  ────────────────────────► dashboard  (badges + EMERGENCY banner)
  ├── /waypoint_manager/goal  (empty = cancel on EMERGENCY_STOP)  ► WM
  └── /jetauto_controller/cmd_vel  ───────────────► JetAuto base

PS  (perception_supervisor_node.py)
  └── /perception/status  ──────────────────────────► dashboard  (badge + start/stop buttons)
        ▲ /perception/cmd                              runs ./run-models.sh | ./stop-models.sh

Dashboard (browser, rosbridge :9090)
  ├── publishes ► /waypoint_manager/goal
  ├── publishes ► /remap_transform
  ├── publishes ► /lane_controller/enable  (FERMA button emergency stop)
  ├── publishes ► /perception/cmd          (start/stop the models)
  └── publishes ► /bev_calibration/cmd     (BEV calibration panel)
```
