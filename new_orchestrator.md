# New Orchestrator — disagreement-driven blend + map-aware roundabout

Experimental control logic in `jetauto_autonomous/scripts/new_orchestrator.py`. It is a
**subclass** of the working `orchestrator.py`, so it inherits all the proven machinery
(remap, drift correction, pure-pursuit `_carrot`/`_pursuit_twist`, callbacks, publishers,
in-place junction spin) and overrides only the per-tick decision logic. Run it *in place of*
`orchestrator.py` (never both — same node name). No other control file is required; new
parameters all have in-code defaults and are also exposed in `lane_params.yaml`.

## Why this exists

The previous orchestrator drove well but failed two tests structurally:

- **Missed a junction turn (went straight).** The map had *zero* influence until the robot
  was within `junction_radius` of the node, so confident lane detection drove straight
  through before the turn could trigger.
- **Cut the roundabout / cut its exit.** Pure-pursuit on the sparse ring nodes aimed across
  chords, and the exit handoff to the lane followed the ring line off-road.

The blend, the radial ring curve, the exit handling, and a terminal safety stop below target
exactly those failures.

## Notation & conventions

- `l` — lane-detection direction. The lane node only publishes `angular.z` (a steering rate),
  so `l` is a **proxy**: `theta_l = clamp(angular.z / map_max_angular, -1, 1) * proxy_max`.
- `m` — map direction: `theta_m = angle_diff(heading_to_next, robot_yaw)`.
- `c` — cosine similarity `cos(theta_l - theta_m)`. `c≈1` agree, `c<0` differ by >90°.
- `a_lane` — blend weight on **lane**: `out = a_lane*lane + (1-a_lane)*map`. `a_lane=1` pure
  lane, `a_lane=0` pure map.
- `c` only *selects* `a_lane`; the blend itself stays on the scalar `angular.z`.
- EMA / drift-correction / remap logic unchanged — inherited from the parent.

## States (one handler each)

`IDLE · NAVIGATING · JUNCTION · ROUNDABOUT · FALLBACK · EMERGENCY_STOP · DONE`

### NAVIGATING (straights & curves)
```
1. c < 0 for conflict_ticks consecutive ticks  -> EMERGENCY_STOP   (debounced)
     SKIPPED while the current node is a junction: map heading points at the
     POST-junction node, so lane(straight) vs map(into turn) are *meant* to
     disagree there. A genuine wrong-way still trips once past the junction.
     ALSO gated on being ON the path (cross-track <= conflict_onpath_m): when the
     robot has drifted off-route, the bearing-to-next-node is large so c collapses
     to cos(theta_m) and a straight, recovering lane reads as a >90deg "conflict"
     even though nothing is wrong-way. That excursion is a recovery FALLBACK owns;
     off the path the map heading is not a trustworthy "this is the way" signal.
2. Sharp junction within junction_radius        -> JUNCTION         (in-place spin)
3. Lane HOLD/STOP for hold_ramp ticks            -> FALLBACK         (pure-pursuit)
4. else (on route):
     a_lane = 1 on open road (pure lane).
     If the current node is a junction, a_lane is distance-gated over
     junction_influence_radius so the map starts pulling into the turn before
     the node (far = trust lane, near + lane disagrees = map takes over).
```
*(An opt-in off-route REPLAN exists — `offroute_enable`, default `False` — that only
republishes the goal for a fresh Dijkstra; it never seizes steering. Left off by default.)*

### JUNCTION (in-place spin)
`m` targets the node *after* the junction. The robot rotates toward `heading_to_next`; when
aligned within `junction_align_deg` it applies a theta drift-fix and returns to NAVIGATING.
Re-trigger is guarded per node (`_handled_junction`). Gentle turns (< `gentle_turn_deg`) skip
the spin and stay on the blend.

### ROUNDABOUT (radial ring + pure-pursuit exit + camera guardrail)
Entered when `map.is_roundabout_node(cur)` or `(next)` **or** `_is_exit_stub(cur)`. Lane is
unreliable here (the solid centre island reads as a short lane segment, the outer line runs
straight out the entrance road), so the **map drives**; the camera only guards the edges.

Two phases, chosen per tick by whether the **next** target is still a ring node:

- **On the ring** (`next` is a ring node): the carrot rides a per-segment **radial arc**
  through the ring nodes (`radial_ring_curve`, built once and cached, keyed on the ring node
  ids; centre = centroid of the ring nodes). Each ring segment is a polar arc keeping each
  node's own radius, so an outlying node → flatter local arc, a tighter node → sharper arc;
  the curve never bulges past the node radii (no chord-cutting). Falls back to node
  pure-pursuit when < 3 ring nodes.
- **On the exit approach** (`next` has LEFT the ring, e.g. `28→23→2`): the spline is
  **dropped** for plain pure-pursuit (`_carrot`) straight at the exit node. The spline's
  tangent at the last ring node points *around* the ring, so riding it there made the robot
  sail past the exit and then cut the `28→23` edge ~90° off-road. `_carrot` aims into the
  exit from the start (smooth turn-in) and recomputes from the nearest live path node every
  tick, so it cannot latch onto a stale curve endpoint and run away.

`angular.z = base(carrot) + guardrail_nudge`; `linear.x = roundabout_drive_speed`.

**Exit authority (`_is_exit_stub`).** The roundabout window is extended through the **exit
stub** — the first non-ring node whose path predecessor is a ring node (e.g. 23 reached from
28). The map keeps authority across it so the lane controller can't grab the `28→23` exit
drive and follow the ring line off-road; the lane resumes only once the predecessor is no
longer a ring node (genuinely out, e.g. at node 2).

**Camera guardrail (`_round_lane_nudge`).** Uses `/lane_controller/info` (camera-frame lane
geometry). NOT lane-following — returns 0 unless a painted line is *confidently seen* and
close to the robot centre:
- OUTER (road-edge) line too close → nudge **inward**;
- INNER (island) line too close → nudge **outward**.
Inner/outer is decided per tick from which side the ring centre is on (CW/CCW-agnostic). It
returns 0 on the whole exit approach (`next` not a ring node), so the exit drives unopposed.
If the relevant line isn't seen, the map curve drives unaided.

**Off-reference failsafe.** If the robot strays more than `roundabout_offref_m` from its own
reference (the spline on the ring, the path on the exit) for `roundabout_offref_ticks`
consecutive ticks → **EMERGENCY_STOP**. Lane is unreliable here, so this geometric check is
the only roundabout safety net. A lenient lane/map conflict (`c < roundabout_conflict_c`,
debounced) also trips it.

### FALLBACK
Pure-pursuit on the path; recover to NAVIGATING when lane-OK **and** `dist ≤ recovery_radius`
for `recovery_ticks` (inherited hysteresis).

### EMERGENCY_STOP (terminal failsafe)
A single terminal halt shared by every failsafe trigger (NAVIGATING conflict, roundabout
conflict, roundabout off-reference). On entry it:
- disables the lane controller and halts the wheels (zero `Twist`);
- **cancels navigation** by publishing an empty goal to `/waypoint_manager/goal` (waypoint
  manager → IDLE, exactly like arrival);
- latches `_emergency` and publishes `EMERGENCY_STOP` on `/orchestrator/state`.

It is held at the top of `_step` (above the inactive/DONE handler, so the nav-cancel can't
mask it). **No auto-recovery** — the debounce upstream already absorbs transient camera /
segmentation noise, so a trip is a sustained fault. Resume only by issuing a **new goal**
from the dashboard (detected in `_path_cb`, which clears the latch and returns to NAVIGATING).
The dashboard shows a pulsing red banner + `⛔ EMERGENCY STOP` badge; because the lane is
disabled, the drive indicator also flips to FERMA.

## Parameters (in-code defaults; all also in `lane_params.yaml` `orchestrator:`)

| Param | Default | Meaning |
|---|---|---|
| `conflict_ticks` | 12 | consecutive `c<0` ticks before EMERGENCY_STOP |
| `conflict_onpath_m` | 0.35 m | only judge a conflict while cross-track to the path is below this (off-route, `c` collapses to `cos(theta_m)` and false-trips on a recovering lane) |
| `proxy_max_deg` | 80 | full lane steer → this implied heading offset |
| `junction_influence_radius` | 0.50 m | distance over which the map blends into a turn; MUST exceed `junction_radius` (0.30 missed turns on an imperfect map) |
| `roundabout_drive_speed` | 0.10 m/s | map-following speed inside the roundabout |
| `roundabout_lookahead_m` | 0.25 | carrot lookahead along the ring spline (exit uses `lookahead_m`=0.50) |
| `roundabout_spline_res_m` | 0.03 | sample spacing of the dense ring curve |
| `roundabout_conflict_c` | -0.5 | lenient conflict threshold inside the roundabout |
| `roundabout_lane_correct` | True | master enable for the camera guardrail nudge |
| `roundabout_outer_clear` | 0.25 | normalized clearance to the OUTER line below which we nudge inward |
| `roundabout_inner_clear` | 0.20 | normalized clearance to the INNER line below which we nudge outward |
| `roundabout_lane_gain` | 0.40 | rad/s per unit edge-proximity severity |
| `roundabout_lane_max` | 0.50 | rad/s cap on the guardrail nudge |
| `roundabout_offref_m` | 0.40 m | deviation from the reference before EMERGENCY_STOP |
| `roundabout_offref_ticks` | 10 | consecutive off-reference ticks (debounce) |

Reused from the parent: `junction_radius`, `junction_align_deg`, `junction_spin_speed`,
`junction_alpha`, `gentle_turn_deg`, `hold_ramp_ticks`, `recovery_ticks`,
`fallback_recovery_radius`, `lookahead_m`, `map_kp`, `map_max_angular`, drift params.

### `/lane_controller/info` (Float64MultiArray, additive — published by `lane_controller_node.py`)
Does **not** touch the existing `cmd_vel`/`state` path. Consumed by the roundabout guardrail.
```
[0] state_code (0 STOP,1 HOLD,2 TRACKING_CC,3 SINGLE_L,4 SINGLE_R,5 DISABLED)
[1] heading_rad (real lane direction vs robot forward; +ve bends right)
[2] left_valid  [3] right_valid
[4] left_offset [5] right_offset [6] center_offset   (normalized to W/2; |.|~0 = line at robot centre)
[7] lane_width  (normalized to W/2)
```
`heading_rad` is the real lane vector — a future cleanup can replace the `angular.z` proxy
for `c` with it. For now only the offsets are consumed (roundabout guardrail only).

## Running it

```bash
# stack as usual, but launch the new node instead of orchestrator.py
rosrun jetauto_autonomous new_orchestrator.py
```

`decide_blend(...)` and `radial_ring_curve(...)` are pure functions (no ROS) and can be
unit-tested or wired into `Testing/offline_tester.py`.