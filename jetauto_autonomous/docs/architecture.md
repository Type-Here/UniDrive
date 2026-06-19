# System Architecture — JetAuto Autonomous Driving

This document describes how the autonomous driving stack works end-to-end: from raw camera pixels to motor commands. It is intended as an internal reference for understanding, debugging, and extending the system.

---

## 1. Two Python environments

The stack is split across two completely separate Python processes that communicate only via ROS topics.

| Process | Interpreter | Location | What it does |
|---|---|---|---|
| `perception_node.py` | Python 3 (conda) | `jetauto_autonomous/perception/` | Merged perception node: YOLO11/TensorRT object detection (traffic lights, STOP signs) **and** lane segmentation (ONNX/TensorRT). Launched with `./run-models.sh` |
| Everything else | Python 2.7 (system ROS) | `jetauto_autonomous/scripts/` | All control logic, waypoint management, orchestration |

> **Superseded:** `on_jetauto_scripts/drive_segm/lane_follower.py` was the original segmentation-only node. It is kept in the repository as a reference but replaced on-robot by `perception_node.py`.

They can never share objects or function calls — the boundary is always a ROS topic message.

---

## 2. Full data-flow pipeline

```
Camera
  /depth_cam/rgb/image_raw (640×480, rgb8)
        │
        ▼
  perception_node.py  ── Python 3, ONNX/TensorRT  (--mode both)
  (detection + segmentation, launched via ./run-models.sh)
        │
        ├──► /lane_mask_bev    (BEV-warped mask, mono8)          -- segmentation
        │               │
        │               ▼
        │   lane_controller_node.py  ── Python 2.7
        │   (HoughLinesP → polynomial fit → adaptive EMA steering)
        │               │
        │               ├──► /lane_controller/cmd_vel   (proposed Twist)
        │               └──► /lane_controller/state     (OK / HOLD / STOP / DISABLED)
        │                                │
        └──► /object_detection/drive  (JSON detections)  -- detection
             (TRAFFIC_STOP / STOP_SIGN override)
                    ┌───────────────────┘
                    │          /odom
                    │            │
                    ▼            ▼
             n_orchestrator.py  ── Python 2.7
             (FSM + blending + junction + roundabout + fallback)
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

**Key design principle:** the orchestrator (`n_orchestrator.py`) is the *sole* publisher of `/jetauto_controller/cmd_vel`. No other node touches the hardware command topic. This eliminates the race condition that previously occurred when `lane_controller_node` and `waypoint_manager_node` both published simultaneously during state transitions.

`n_orchestrator.py` is self-contained: the former base class `Orchestrator` (remap, drift fix, pure-pursuit, junction spin) is defined inline in the same file and `NewOrchestrator` extends it. The legacy `orchestrator.py` and `new_orchestrator.py` (superseded experiment) have been removed.

---

## 3. Node-by-node description

### 3.1 `perception_node.py` (Python 3)

The merged perception node — runs in `--mode both` on the robot (one camera subscriber, one CUDA context, two models). It is **perception-only**: it **never** publishes a velocity command.

Launched via `./run-models.sh` from `jetauto_autonomous/`; lives in `jetauto_autonomous/perception/`. TensorRT engines are placed in `jetauto_autonomous/perception/models/`.

**What it outputs:**
- `/lane_mask_bev` — the primary input to the control stack. A 320×128 bird's-eye-view binary mask, class labels: 0=background, 1=road, 2=lane_marking, 3=lane_dashed, 4=zebra. The BEV warp is calibrated on first run, then cached in `perception/calibration.json`.
- `/object_detection/drive` — JSON `{"detections": [{class_name, score, box}]}`, consumed by the orchestrator's traffic-light/STOP override.
- `/object_detection/video` — annotated detection overlay (optional, for the dashboard).
- `/lane_follower/debug_image` — coloured segmentation overlay + BEV mask (when `--debug`, enabled by default).

**Modes:**
- `--mode both` (default): detection + segmentation in the same process/CUDA context.
- `--mode detection`: object detection only (no `/lane_mask_bev` published).
- `--mode segmentation`: lane segmentation only (no `/object_detection/*` published).

> **Superseded:** `on_jetauto_scripts/drive_segm/lane_follower.py` was the original segmentation-only node, kept for reference only.

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
2. Dot-product test on path segments, measured along the **incoming** edge: if `(robot - node_i) · (node_i - node_{i-1}) > 0`, the robot has traveled *through* `node_i` → advance idx. (The outgoing edge was used originally, but at a sharp junction it points sideways, so a small lateral map error advanced the target before the robot ever reached the node and the turn never armed.)
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

### 3.4 `n_orchestrator.py` (Python 2.7)

The brain of the driving system. It is the only node that publishes to `/jetauto_controller/cmd_vel`. It is self-contained: it defines a base `Orchestrator` class inline (the remap machinery, drift correction, pure-pursuit and the junction spin) and `NewOrchestrator` extends it, overriding the per-tick decision logic — see the dedicated orchestrator doc for the full algorithm (blend, roundabout phases, failsafes).

**Inputs:**
- `/lane_controller/cmd_vel` — what lane detection wants to do
- `/lane_controller/state` — how confident lane detection is
- `/lane_controller/info` — camera-frame lane geometry (heading, per-line offsets/validity, dashed flags)
- `/waypoint_manager/nav_info` — where on the map we are
- `/waypoint_manager/path` — full path node IDs (for pure-pursuit)
- `/odom` — robot pose (converted to MAP frame internally)
- `/remap_transform` — coordinate frame update
- `/object_detection/drive` — perception node's JSON detections, consumed by the
  `object_detection.traffic_sign_handler.TrafficSignHandler` (called via function each tick;
  red light / STOP sign → full-stop override). Set `traffic_light_enable: false` to ignore.

**Outputs:**
- `/jetauto_controller/cmd_vel` — the actual motor command
- `/lane_controller/enable` — controls whether lane_controller runs its Hough loop
- `/orchestrator/state` — FSM state for the dashboard (incl. `EMERGENCY_STOP`, plus the
  transient object-detection overrides `TRAFFIC_STOP` / `STOP_SIGN`)
- `/orchestrator/diag` — 13-field per-tick diagnostics (cross-track, blend, remap, …)
- `/remap_transform` — re-published when the drift/straight corrections adjust the frame
- `/waypoint_manager/goal` — empty goal to cancel navigation on emergency stop (and opt-in replan)

---

### 3.5 Startup sequence (updated)

`start_all.sh` starts the Python 2.7 control stack (rosbridge, video server, dashboard, lane controller, waypoint manager, orchestrator). Then, in a separate conda terminal:

```bash
cd jetauto_autonomous
./run-models.sh           # starts perception_node.py --mode both in the background
./stop-models.sh          # graceful shutdown
```

The orchestrator-side consumer of `/object_detection/drive` is the small Python 2.7 module `jetauto_autonomous/scripts/object_detection/traffic_sign_handler.py`, called via function each tick. Set `object_detection_enable: false` in `lane_params.yaml` to ignore detections and run the driving stack on its own.

---

## 4. Orchestrator FSM

```
     ┌──────┐    new goal (is_active = True)    ┌───────────┐
     │ IDLE │ ─────────────────────────────────► │ NAVIGATING│
     └──────┘                                   └─────┬─────┘
                                                      │
         ┌──────────────────────┬─────────────────────┤
         │ sharp junction AND   │ cur/next is a ring  │ HOLD/STOP ticks
         │ dist ≤ radius        │ node (or exit stub) │ ≥ hold_ramp
         ▼                      ▼                     ▼
   ┌──────────┐          ┌────────────┐         ┌──────────┐
   │ JUNCTION │          │ ROUNDABOUT │         │ FALLBACK │
   └────┬─────┘          └─────┬──────┘         └────┬─────┘
        │ aligned              │ window closed       │ lane OK near path
        └──────────────────────┴─────────────────────┘
                               │
                               ▼  back to NAVIGATING
                               │
                               │ nav_info.is_active = False (goal reached or canceled)
                               ▼
                            ┌──────┐
                            │ DONE │
                            └──────┘

   any sustained failsafe ──────────────────► ┌────────────────┐
   (lane/map conflict, roundabout or          │ EMERGENCY_STOP │
    fallback off-reference, fallback timeout) └────────────────┘
                                              terminal: halts, cancels the goal;
                                              cleared only by a NEW goal
```

---

## 5. Driving decisions in detail

> **Note** — this section describes the *base-class* logic (the `Orchestrator` class now
> defined inline in `scripts/n_orchestrator.py`) that the FSM is built on. The running
> `NewOrchestrator` in the same file replaces the alpha blend of §5.1
> with a disagreement-driven `a_lane` blend, adds the ROUNDABOUT state (radial ring curve +
> camera guardrail), the EMERGENCY_STOP failsafes, and continuous map-frame drift corrections.
> See the dedicated orchestrator doc for the current algorithm; the JUNCTION and FALLBACK
> mechanics below still apply.

### 5.1 NAVIGATING — blended lane + map (base class)

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
7. `n_orchestrator.py` — starts in IDLE, immediately publishes `enable=False` to lane_controller

Then, separately, in a conda terminal:
```bash
cd jetauto_autonomous
./run-models.sh           # both models (detection + segmentation)
# ./run-models.sh detection     # detection only (no /lane_mask_bev)
# ./run-models.sh segmentation  # segmentation only (no /object_detection/*)
```

To disable traffic-light / STOP handling, set `object_detection_enable: false` in
`lane_params.yaml` and restart the orchestrator (or run `--mode segmentation`).

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
| `/depth_cam/rgb/image_raw` | Image | hardware | perception_node | 640×480 |
| `/lane_mask_bev` | Image mono8 | perception_node | lane_controller | BEV warped, 320×128 |
| `/lane_controller/cmd_vel` | Twist | lane_controller | orchestrator | **proposed only, never to hardware** |
| `/lane_controller/state` | String | lane_controller | orchestrator | OK/HOLD/STOP/DISABLED |
| `/lane_controller/enable` | Bool (latched) | orchestrator | lane_controller | sole enable publisher |
| `/lane_controller/info` | Float64MultiArray | lane_controller | orchestrator | 10-field lane geometry (heading, line offsets/validity, dashed flags) |
| `/lane_debug/image` | Image | lane_controller | dashboard | debug overlay |
| `/odom` | Odometry | hardware | waypoint_manager, orchestrator | robot pose |
| `/remap_transform` | Float64MultiArray | dashboard, orchestrator | waypoint_manager, orchestrator | live frame update (orchestrator republishes its drift corrections) |
| `/waypoint_manager/goal` | Int32MultiArray | dashboard, orchestrator | waypoint_manager | [start_id, end_id]; **empty = cancel** (used by the emergency stop) |
| `/object_detection/drive` | String (JSON) | perception_node | orchestrator | detections; drives the `TRAFFIC_STOP` / `STOP_SIGN` override |
| `/object_detection/video` | Image bgr8 | perception_node | dashboard | annotated detection overlay (optional) |
| `/orchestrator/state` | String (latched) | orchestrator | dashboard | FSM state, incl. `EMERGENCY_STOP` + transient `TRAFFIC_STOP` / `STOP_SIGN` |
| `/orchestrator/diag` | Float64MultiArray | orchestrator | logging / plots | 13-field per-tick diagnostics (25 Hz) |
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
| `traffic_light_enable` | `true` | Object-detection override on/off (`false` = ignore detections, driving-only) |
| `stop_sign_hold` | 3.0 s | How long to hold at a detected STOP sign |

---

## 10. Known limitations and discussion points

- **Junction advancement race:** after the orchestrator completes rotation and creeps forward, the waypoint_manager's dot-product test advances `idx` past the junction node. There is a brief period (~1–3 ticks) where `idx` still points at the junction node while the robot is already moving away. The `junction_radius` check in the orchestrator has a sticky `self._state == JUNCTION` guard that prevents re-triggering, so this is safe.

- **Junction passing deadlock (open issue):** the advancement test measures progress along the *incoming* edge. If the junction turn happens *before* the node's perpendicular plane (the spin arms up to `junction_radius` early, or the blend cuts the corner), all subsequent travel is perpendicular to the test axis and a small lateral map offset can keep the test from ever firing — `idx` then stays parked on the junction node (observed at node 6: target stuck at `6->30`, `dist` growing, `c → -1` once past node 30). While stuck, the conflict stop is suppressed (`is_junction`) and the straight-line corrections are gated off, so the condition cannot self-heal. A related trigger: a lateral map offset can keep `dist` above `junction_radius` so the in-place spin never arms and the blend alone cuts the turn.

- **Roundabout tuning:** `alpha_near_junction` applies whenever `is_junction=True`, which depends on the node degree in the map graph. If the roundabout arc nodes are degree-2 (in/out only), they are not flagged as junctions and α stays at 0. In that case the blend only helps at the entry/exit nodes. The fix is to mark arc nodes differently in the map, or add a dedicated node type.

- **Pure-pursuit in FALLBACK uses the full path:** `_carrot()` finds the closest node on the entire path (not just ahead). If the robot has already advanced past several nodes and then falls back to FALLBACK, the closest-node search still finds the right segment because waypoint_manager's dot-product has already advanced `idx` — but the orchestrator doesn't know `idx`. For most scenarios this works fine; in pathological cases (robot went backwards) it might pick a behind-robot carrot. If this becomes an issue, pass `nav_info[_NI_IDX]` to `_carrot()` to slice the path.

- **No explicit DONE state publisher from orchestrator:** the orchestrator transitions to DONE when `nav_info.is_active` drops to 0 (i.e., waypoint_manager sets GOAL_REACHED). The dashboard reads `/waypoint_manager/status` directly for this signal, which is correct.
