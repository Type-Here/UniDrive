# System Architecture — JetAuto Autonomous Driving

This document describes how the autonomous driving stack works end-to-end: from raw camera pixels to motor commands. It is intended as an internal reference for understanding, debugging, and extending the system.

---

## 1. Two Python environments

The stack is split across two completely separate Python processes that communicate only via ROS topics.

| Process | Interpreter | Location | What it does |
|---|---|---|---|
| `lane_follower.py` | Python 3 (conda) | `on_jetauto_scripts/drive_segm/` | Neural network inference (ONNX / TensorRT); publishes segmentation masks |
| Everything else | Python 2.7 (system ROS) | `jetauto_autonomous/scripts/` | All control logic, waypoint management, orchestration |

They can never share objects or function calls — the boundary is always a ROS topic message.

---

## 2. Full data-flow pipeline

```
Camera
  /depth_cam/rgb/image_raw (640×480, rgb8)
        │
        ▼
  lane_follower.py  ── Python 3, ONNX/TensorRT
  (segmentation-only)
        │
        └──► /lane_mask_bev    (BEV-warped mask, mono8)
                    │
                    ▼
        lane_controller_node.py  ── Python 2.7
        (HoughLinesP → polynomial fit → adaptive EMA steering)
                    │
                    ├──► /lane_controller/cmd_vel   (proposed Twist)
                    └──► /lane_controller/state     (OK / HOLD / STOP / DISABLED)
                                     │
                    ┌────────────────┘
                    │          /odom
                    │            │
                    ▼            ▼
             orchestrator.py  ── Python 2.7
             (FSM + blending + junction + fallback)
                    │
                    ├──► /lane_controller/enable    (enable/disable Hough loop)
                    └──► /jetauto_controller/cmd_vel  ◄── sole hardware output
                                     │
                    ┌────────────────┘
                    ▼
              JetAuto base controller
              (mecanum wheels)

  waypoint_manager_node.py  ── Python 2.7
  (Dijkstra + dot-product path tracking)
        │
        └──► /waypoint_manager/nav_info  ──► orchestrator.py
        └──► /waypoint_manager/path      ──► orchestrator.py
        └──► /waypoint_manager/status    ──► dashboard
```

**Key design principle:** `orchestrator.py` is the *sole* publisher of `/jetauto_controller/cmd_vel`. No other node touches the hardware command topic. This eliminates the race condition that previously occurred when `lane_controller_node` and `waypoint_manager_node` both published simultaneously during state transitions.

---

## 3. Node-by-node description

### 3.1 `lane_follower.py` (Python 3)

Runs the neural network. It is segmentation-only: it **never** publishes a velocity command — its only job is to produce the BEV segmentation mask.

**What it outputs:**
- `/lane_mask_bev` — the primary input to the control stack. A 320×128 (or scaled) bird's-eye-view binary mask where pixels are labelled by class: 0=background, 1=road, 2=lane_marking, 3=lane_dashed, 4=zebra.

The BEV warp is calibrated once on first run (interactive corner selection), then cached in `calibration.json`.

---

### 3.2 `lane_controller_node.py` (Python 2.7)

Converts the mask into a steering command. All pure logic lives in `lane_core.py` (no ROS imports there); this file only does ROS wiring.

**Pipeline per frame:**
1. Receive `/lane_mask_bev`
2. Optionally upscale the mask by `bev_scale` (default ×2 → 640×256)
3. Crop to ROI (top `hough_roi_top_frac` discarded)
4. `HoughLinesP` → classify lines as left/right by slope
5. Polynomial fit (linear or quadratic) → find left and right lane edges
6. Compute centre offset at `center_y_ratio` height
7. Convert offset to steering angle (degrees), capped at `max_steering_angle`
8. Adaptive EMA smoothing: α increases with angular velocity to be more responsive during sharp turns
9. Convert angle to `angular.z` and publish

**States published on `/lane_controller/state`:**

| State | Meaning |
|---|---|
| `TRACKING_CC` | Both lines visible (centre-centre measurement) |
| `TRACKING_DC` | Both lines visible (dynamic-width centre) |
| `SINGLE_L` / `SINGLE_R` | Only one line visible; other inferred from lane width |
| `HOLD` | No valid lines this frame; holds last steering angle |
| `STOP` | No lines for several frames; publishes zero |
| `DISABLED` | Node received `enable=False` |

**Output topic:** `/lane_controller/cmd_vel` — a proposed Twist. This is **not** sent to hardware; it is consumed by the orchestrator.

The node also subscribes to `/lane_controller/enable` (Bool, latched). When `False`, it stops processing and publishes DISABLED state; this reduces CPU load (Hough is not run during junction rotation or full map fallback).

---

### 3.3 `waypoint_manager_node.py` (Python 2.7)

Pure path-tracker. Knows the map but makes no driving decisions.

**On startup:** loads the map YAML (a NetworkX directed graph of nodes with (x, y) coordinates and type tags). Runs Dijkstra when a goal arrives.

**Each tick:**
1. Read current robot pose (MAP frame, via odom + remap transform)
2. Dot-product test on path segments: if `(robot - node_i) · (node_{i+1} - node_i) > 0`, the robot has passed `node_i` → advance idx
3. Final node: stop if distance < `waypoint_tolerance`
4. Publish `/waypoint_manager/nav_info` (8-element Float64MultiArray):

```
[0] current_node_id
[1] next_node_id         (-1 if last node)
[2] dist_to_current      (metres)
[3] is_junction          (1.0 if current node is a junction, else 0.0)
[4] heading_to_next      (radians in MAP frame: bearing from robot to next node)
[5] path_idx             (current index in path array)
[6] path_len             (total nodes in path)
[7] is_active            (1.0 if navigating, 0.0 if idle/done)
```

The **remap transform** (theta, scale, tx, ty) converts odom coordinates to the map's coordinate frame. It is loaded from `web/remap_params.json` at startup and updated live via `/remap_transform` when the user recalibrates via the dashboard.

---

### 3.4 `orchestrator.py` (Python 2.7)

The brain of the driving system. It is the only node that publishes to `/jetauto_controller/cmd_vel`.

**Inputs:**
- `/lane_controller/cmd_vel` — what lane detection wants to do
- `/lane_controller/state` — how confident lane detection is
- `/waypoint_manager/nav_info` — where on the map we are
- `/waypoint_manager/path` — full path node IDs (for pure-pursuit)
- `/odom` — robot pose (converted to MAP frame internally)
- `/remap_transform` — coordinate frame update

**Outputs:**
- `/jetauto_controller/cmd_vel` — the actual motor command
- `/lane_controller/enable` — controls whether lane_controller runs its Hough loop

---

## 4. Orchestrator FSM

```
         ┌──────────────────────────────────────────┐
         │  nav_info.is_active = False              │
         ▼                                          │
     ┌──────┐    new goal (is_active = True)   ┌───────────┐
     │ IDLE │ ────────────────────────────────► │ NAVIGATING│
     └──────┘                                  └─────┬─────┘
                                                     │
              ┌──────────────────────────────────────┤
              │                                      │
              │  is_junction AND dist ≤ radius       │  HOLD ticks ≥ hold_ramp
              ▼                                      ▼
        ┌──────────┐                          ┌──────────┐
        │ JUNCTION │                          │ FALLBACK │
        └────┬─────┘                          └────┬─────┘
             │                                     │
             │ aligned AND creep done              │ OK ticks ≥ recovery_ticks
             └──────────────┐  ┌──────────────────┘
                            ▼  ▼
                       ┌───────────┐
                       │ NAVIGATING│
                       └─────┬─────┘
                             │
                             │  nav_info.is_active = False (goal reached or canceled)
                             ▼
                          ┌──────┐
                          │ DONE │
                          └──────┘
```

---

## 5. Driving decisions in detail

### 5.1 NAVIGATING — blended lane + map

This is the default state. The orchestrator computes a blended command:

```
angular.z_out = (1 - α) * lane_angular + α * map_angular
linear.x_out  = lane_linear.x
```

`map_angular` is a P-controller toward the map heading:
```
map_angular = clamp(kp * angle_diff(heading_to_next, robot_yaw), -max_w, +max_w)
```

**Dynamic alpha (α):**

| Condition | α | Robot behaviour |
|---|---|---|
| `lane_state = OK or TRACKING_*`, normal node | 0.0 | 100% lane; map ignored |
| `lane_state = OK`, current node is junction type | `alpha_near_junction` (default 0.6) | Map guardrail active; dampens wrong lane reactions (e.g. roundabout centre circle) |
| `lane_state = HOLD`, tick `t` (0 → hold_ramp-1) | `t / hold_ramp_ticks` | Smooth ramp from lane toward map as confidence drops |
| `lane_state = HOLD` for ≥ `hold_ramp_ticks` | → FALLBACK | Full map takeover |
| Final segment (no next node) | 0.0 | Pure lane until distance stop |

Speed (`linear.x`) always comes from the lane controller, except in FALLBACK where `map_drive_speed` is used.

**Why this works for roundabouts:** the roundabout entry/exit nodes are typically junction-type nodes (degree > 2 in the graph). As the robot approaches, `α = alpha_near_junction` blends the map heading into the steering command, suppressing the erratic reaction to the central circle without completely ignoring the lane markings.

### 5.2 JUNCTION — heading-based rotation

Triggered when the current target node is a junction AND the robot is within `junction_radius` metres of it AND there is a next node.

**Behaviour:**
1. Disable lane controller (`enable=False`) — stops Hough processing
2. Compute `err = angle_diff(heading_to_next, robot_yaw)` in MAP frame
3. If `|err| ≥ junction_align_deg`: spin in-place — `angular.z = clamp(1.5 × err, ±junction_spin_speed)`
4. If `|err| < junction_align_deg`: creep forward at `junction_creep_speed` for one tick, re-enable lane controller, transition to NAVIGATING

After re-enabling, the dot-product test in waypoint_manager will advance idx past the junction node once the robot has moved forward enough.

### 5.3 FALLBACK — pure-pursuit on map

Triggered when `lane_state ∈ {HOLD, STOP}` for `hold_ramp_ticks` consecutive ticks (default 15 × 40ms = 600ms).

**Behaviour:** pure-pursuit controller
1. Find the closest node on the path
2. Walk forward along path nodes accumulating distance until `lookahead_m` is reached → this is the carrot point
3. Compute heading from robot to carrot → angular error → `angular.z`
4. Forward speed: `map_drive_speed` (slower than normal to compensate for reduced accuracy)

**Recovery:** when `lane_state` returns to a non-bad state (OK / TRACKING / etc.) for `recovery_ticks` consecutive ticks (default 5 × 40ms = 200ms), the orchestrator re-enables lane controller and returns to NAVIGATING. The hysteresis window prevents thrashing when lane detection flickers.

---

## 6. Coordinate frames

The system uses two frames:

| Frame | Origin | Used by |
|---|---|---|
| **odom** | Robot starting pose | `/odom` publisher (Hiwonder driver), lane_controller |
| **map** | Map YAML origin | Map node coordinates, waypoint_manager, orchestrator |

The remap transform `(theta, scale, tx, ty)` converts between them:
```
map_xy  = R(-theta) / scale × (odom_xy - [tx, ty])
odom_yaw → map_yaw = odom_yaw - theta
```

Both `waypoint_manager_node` and `orchestrator` apply this transform independently (they both load `remap_params.json` at startup and subscribe to `/remap_transform` for live updates). This is intentional redundancy — the orchestrator needs map-frame coordinates for pure-pursuit and junction checks without depending on waypoint_manager for pose data.

---

## 7. Startup sequence

`start_all.sh` starts processes in this order (each waits 0.5s before checking the PID):

1. `rosparam load config/lane_params.yaml` — all parameters available before any node starts
2. `rosbridge_websocket :9090` — dashboard WebSocket bridge
3. `web_video_server :8080` — MJPEG image streaming
4. `serve_dashboard.py :8000` — HTTP dashboard server
5. `lane_controller_node.py` — starts in DISABLED state (enable not yet published)
6. `waypoint_manager_node.py` — starts in IDLE, publishes `nav_info.is_active=False`
7. `orchestrator.py` — starts in IDLE, immediately publishes `enable=False` to lane_controller

Then, separately, in a conda terminal:
```bash
python3 lane_follower.py --model model.engine --tensorrt
```

**What happens when a goal is sent from the dashboard:**
1. Dashboard publishes `[start_id, end_id]` to `/waypoint_manager/goal`
2. `waypoint_manager` runs Dijkstra, publishes path, starts publishing `nav_info` with `is_active=1.0`
3. `orchestrator` sees `is_active=1.0`, transitions to NAVIGATING, publishes `enable=True`
4. `lane_controller` receives `enable=True`, starts processing masks, publishes proposed `cmd_vel`
5. `orchestrator` receives the lane proposal, blends with map heading (α=0 on normal road), forwards to hardware
6. Robot drives

---

## 8. Topic reference (current)

| Topic | Type | Publisher | Subscribers | Notes |
|---|---|---|---|---|
| `/depth_cam/rgb/image_raw` | Image | hardware | lane_follower | 640×480 |
| `/lane_mask_bev` | Image mono8 | lane_follower | lane_controller | BEV warped, 320×128 |
| `/lane_controller/cmd_vel` | Twist | lane_controller | orchestrator | **proposed only, never to hardware** |
| `/lane_controller/state` | String | lane_controller | orchestrator | OK/HOLD/STOP/DISABLED |
| `/lane_controller/enable` | Bool (latched) | orchestrator | lane_controller | sole enable publisher |
| `/lane_debug/image` | Image | lane_controller | dashboard | debug overlay |
| `/odom` | Odometry | hardware | waypoint_manager, orchestrator | robot pose |
| `/remap_transform` | Float64MultiArray | dashboard | waypoint_manager, orchestrator | live frame update |
| `/waypoint_manager/goal` | Int32MultiArray | dashboard | waypoint_manager | [start_id, end_id] |
| `/waypoint_manager/path` | Int32MultiArray (latched) | waypoint_manager | orchestrator | node IDs for pure-pursuit |
| `/waypoint_manager/nav_info` | Float64MultiArray | waypoint_manager | orchestrator | 8-element, 25 Hz |
| `/waypoint_manager/status` | String (latched) | waypoint_manager | dashboard | IDLE/NAVIGATING/GOAL_REACHED/ERROR |
| `/jetauto_controller/cmd_vel` | Twist | **orchestrator only** | hardware | sole hardware output |

---

## 9. Key tunable parameters (`lane_params.yaml`)

### Lane controller

| Parameter | Default | Effect |
|---|---|---|
| `linear_x_speed` | 0.15 m/s | Forward cruise speed |
| `max_angular_z` | 0.90 rad/s | Max steering speed |
| `bev_scale` | 2.0 | Mask upscale (×1 = 320×128, ×2 = 640×256) |
| `hough_threshold` | 30 | Minimum Hough votes |
| `center_y_ratio` | 0.80 | Measurement height [0=top, 1=bottom] |
| `angle_smooth_alpha_base` | 0.50 | EMA base (lower = smoother) |
| `lane_width_px` | 252 | Static lane width fallback (at bev_scale=1.0) |

### Orchestrator

| Parameter | Default | Effect |
|---|---|---|
| `alpha_near_junction` | 0.6 | Map guardrail strength near junctions (0=off, 1=full map) |
| `hold_ramp_ticks` | 15 | HOLD ticks before α reaches 1.0 and FALLBACK activates |
| `recovery_ticks` | 5 | OK ticks to exit FALLBACK |
| `junction_radius` | 0.30 m | Distance that triggers junction maneuver |
| `junction_align_deg` | 8.0° | Alignment tolerance to exit rotation |
| `junction_spin_speed` | 0.40 rad/s | Max spin speed during rotation |
| `map_drive_speed` | 0.04 m/s | Speed in FALLBACK pure-pursuit |
| `map_kp` | 1.2 | Proportional gain for map heading correction |

---

## 10. Known limitations and discussion points

- **Junction advancement race:** after the orchestrator completes rotation and creeps forward, the waypoint_manager's dot-product test advances `idx` past the junction node. There is a brief period (~1–3 ticks) where `idx` still points at the junction node while the robot is already moving away. The `junction_radius` check in the orchestrator has a sticky `self._state == JUNCTION` guard that prevents re-triggering, so this is safe.

- **Roundabout tuning:** `alpha_near_junction` applies whenever `is_junction=True`, which depends on the node degree in the map graph. If the roundabout arc nodes are degree-2 (in/out only), they are not flagged as junctions and α stays at 0. In that case the blend only helps at the entry/exit nodes. The fix is to mark arc nodes differently in the map, or add a dedicated node type.

- **Pure-pursuit in FALLBACK uses the full path:** `_carrot()` finds the closest node on the entire path (not just ahead). If the robot has already advanced past several nodes and then falls back to FALLBACK, the closest-node search still finds the right segment because waypoint_manager's dot-product has already advanced `idx` — but the orchestrator doesn't know `idx`. For most scenarios this works fine; in pathological cases (robot went backwards) it might pick a behind-robot carrot. If this becomes an issue, pass `nav_info[_NI_IDX]` to `_carrot()` to slice the path.

- **No explicit DONE state publisher from orchestrator:** the orchestrator transitions to DONE when `nav_info.is_active` drops to 0 (i.e., waypoint_manager sets GOAL_REACHED). The dashboard reads `/waypoint_manager/status` directly for this signal, which is correct.
