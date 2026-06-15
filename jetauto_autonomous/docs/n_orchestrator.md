# n_orchestrator — disagreement-driven blend, map-aware roundabout, drift-corrected localization

The production control logic in `jetauto_autonomous/scripts/n_orchestrator.py` — the **sole
publisher of `/jetauto_controller/cmd_vel`**. It is **self-contained**: the base
`Orchestrator` class (remap, drift correction, pure-pursuit `_carrot`/`_pursuit_twist`,
callbacks, publishers, in-place junction spin) is defined **inline in the same file**, and
`NewOrchestrator` extends it, overriding the per-tick decision logic. It supersedes and
replaces the former `old/orchestrator.py` and `old/new_orchestrator.py` (both removed) —
`start_all.sh` launches it via `ORCHESTRATOR_NAME`. Never run two orchestrators at once (same
node name).

New parameters all have in-code defaults; most are also exposed in `lane_params.yaml` under
`orchestrator:` (the ones that are **in-code only** are marked in the table below).

## Why this exists

The original orchestrator drove well but failed structurally on:

- **Missed junction turns (went straight).** The map had zero influence until the robot was
  within `junction_radius` of the node, so confident lane detection drove straight through.
- **Cut the roundabout / its exit.** Pure-pursuit on the sparse ring nodes aimed across
  chords, and the exit handoff let the lane follow the ring line off-road.
- **Fast error accumulation.** Odom yaw drift regenerated lateral error continuously; the
  translation-only drift fix chased a moving target, junctions stopped arming, and runs ended
  with the mapped pose metres from the truth.
- **Unbounded failure states.** FALLBACK could pure-pursue a broken localization off-road
  forever; nothing ever stopped a run that had genuinely gone wrong.

## Notation & conventions

- `l` — lane direction in the robot frame (standard CCW yaw).
  **Primary**: the real lane heading from `/lane_controller/info[1]` (`heading_rad`, image
  convention: positive = lane bends right → **negated**), used whenever the lane state is
  TRACKING_CC / SINGLE_* with ≥ 1 valid fitted line.
  **Fallback** (or `use_lane_heading: false`): the old steering-rate proxy
  `theta_l = clamp(angular.z / map_max_angular, -1, 1) * proxy_max`. The proxy folds
  curvature, centering effort and EMA lag into one number — the real heading measures the
  lane itself, which is why `c` no longer needs compensating gates everywhere.
- `m` — map direction: `theta_m = angle_diff(heading_to_next, robot_yaw)`.
- `c` — cosine similarity `cos(theta_l - theta_m)`. `c≈1` agree, `c<0` differ by >90°.
- `a_lane` — blend weight on **lane**: `out = a_lane*lane + (1-a_lane)*map`. `a_lane=1` pure
  lane, `a_lane=0` pure map. `c` only *selects* `a_lane`; the blend stays on `angular.z`.

**Frame convention** (odom → map, inherited):

```
p_map  = R(-theta)/s · (p_odom - t)        psi_map = psi_odom - theta
```

with remap parameters `theta` (`_remap_theta`), `s` (`_remap_scale`), `t = (tx, ty)` in the
**odom frame**. Every correction uses the inverse `t = p_odom - s·R(theta)·p_map` to re-solve
the translation about an anchor point, so the frame pivots about the **robot** (a theta-only
change would otherwise translate every mapped point by `delta × lever-arm` from the remap
origin — the likely reason the old theta fix was tuned to 0).

## States (one handler each)

`IDLE · NAVIGATING · JUNCTION · ROUNDABOUT · FALLBACK · EMERGENCY_STOP · DONE`

Plus two **transient** object-detection overrides — `TRAFFIC_STOP` / `STOP_SIGN` — that are
*published* on `/orchestrator/state` but never stored in `_state` (see below), so navigation
resumes from wherever it was the instant the override clears.

### NAVIGATING (straights & curves)

```
1. c < 0 for conflict_ticks consecutive ticks  -> EMERGENCY_STOP   (debounced)
     SKIPPED while the current node is a junction: map heading points at the
     POST-junction node, so lane(straight) vs map(into turn) are *meant* to
     disagree there. A genuine wrong-way still trips once past the junction.
     ALSO gated on being ON the path (cross-track <= conflict_onpath_m): off the
     path the bearing-to-next-node is large, c collapses to cos(theta_m), and a
     straight, recovering lane reads as a false "conflict" — that excursion is a
     recovery FALLBACK owns.
2. Sharp junction within junction_radius        -> JUNCTION         (in-place spin)
3. Lane HOLD/STOP for hold_ramp ticks            -> FALLBACK         (pure-pursuit)
4. else (on route):
     a_lane = 1 on open road (pure lane).
     If the current node is a junction, a_lane is distance-gated over
     junction_influence_radius and driven by how hard the MAP wants to turn:
        approach = (infl - dist)/infl
        pull     = max( |theta_m| / turn_full ,  1 - c )
        a_lane   = 1 - approach * pull
     The magnitude term (|theta_m|/turn_full) carries a sharp ~90° crossway:
     there lane and map agree in *direction* (c stays ~0.5) so the old
     cosine-only pull barely engaged and the robot drove straight through. The
     disagreement term (1-c) is kept for the gentle "steering too soon" case.
     NOTE junction_influence_radius must stay BELOW the entry-segment length or
     the blend pulls toward the post-junction heading while still maneuvering
     the previous node (0.90 reached past node 5 onto the opposite straight;
     0.50 ok).
```

*(An opt-in off-route REPLAN exists — `offroute_enable`, default `False` — that only
republishes the goal for a fresh Dijkstra; it never seizes steering.)*

**Anti-cut guardrail (`_junction_anti_cut_scale`).** While a real turn is intended
(`|theta_m| ≥ 20°`) and the line on the INSIDE of the turn is still confidently seen near the
robot centre, the turn is scaled toward straight so the robot waits until the intersection
opens. The scale applies **only to the map term**:

```
w = a_lane*lane_w + (1 - a_lane)*map_w*scale
```

Scaling the whole blend (the old form) also erased the lane's centering correction, leaving
near-zero authority exactly where a command was needed. A **dashed** inside line
(`info[8]/[9]`) is a legally crossable separator — the guardrail returns 1.0 for it.

**Dashed-separator lane change (`_dash_cross_hold`, e.g. node 10→11).** Armed on a junction
approach whose turn points INTO a valid dashed inside line: while the hold runs, `a_lane` is
forced to 0 so the map alone drives the crossing (the lane controller would otherwise
re-center into the original lane after the spin). The tick budget is topped up while the
arming conditions hold, frozen during the JUNCTION spin, and released early once the dashed
line registers on the **opposite** side (= crossed) or after `dash_cross_hold_ticks`.
Dashedness only ever *permits* a crossing; the map decides whether one happens. Log lines:
`dash-cross hold armed/released/expired`.

**Straight-line drift corrections (`_apply_straight_corrections`).** GPS-style re-centering
of the map frame, LATERAL + YAW. When the camera confidently tracks a centred two-line lane
along a straight segment, two things pin the remap simultaneously:

- **Position**: the robot is on the lane centreline = the map edge, so the remap translation
  is nudged perpendicularly onto it (EMA `lateral_correct_alpha`, every
  `lateral_correct_period` NAVIGATING ticks). Lateral **only** (perpendicular foot on the
  segment), so along-track node advancement is untouched.
- **Yaw**: the robot's true map yaw is `seg_yaw + heading_rad`, so `remap_theta` is
  EMA-corrected toward `theta_exact = angle_diff(odom_yaw, seg_yaw + heading_rad)`
  (`lateral_theta_alpha`, deltas above `lateral_theta_max_deg` rejected). This is the half
  the translation-only fix was missing: a yaw error **regenerates** lateral error
  continuously.

The translation is re-solved **after** the theta update about an anchor (the perpendicular
foot when the position part fires, else the current mapped pose with weight 1 — an exact
re-anchor so a yaw-only update doesn't move the pose). Re-published on `/remap_transform` so
the waypoint manager stays in sync. Gate structure (each bug below was found in sim):

| Gate | Detail |
|---|---|
| "On a straight" is judged from the **lane** | `\|heading_rad\| ≤ lateral_align_deg`. Gating on the mapped yaw was a catch-22: once accumulated theta error exceeded the gate, the correction that fixes theta could never fire (19.4° injected drift → 1.5° absorbed). The mapped yaw keeps only a loose 45° wrong-segment sanity bound |
| Cross-track noise floor bounds only the **translation** | yaw runs whenever the lane/segment evidence holds — pinning position while yaw ran away was the old gate order |
| Turn-in-zone block only where the path actually **bends** | turn angle ≥ `gentle_turn_deg` at the current node; an unconditional `dist ≤ junction_influence_radius` gate starved the correction on dense maps (segments 0.30–0.47 m) |
| Plus | NAVIGATING only, never junction/roundabout, TRACKING_CC with both lines valid and `\|center_offset\| ≤ lateral_centered_clear` |

Log: `[n_orch] straight correction: cross=… xy=… dtheta=…` (`xy=0` = yaw-only re-anchor).

### JUNCTION (in-place spin)

`m` targets the node *after* the junction. The robot rotates toward `heading_to_next`; when
aligned within `junction_align_deg` it applies the theta drift-fix and returns to NAVIGATING.
Re-trigger is guarded per node (`_handled_junction`). Gentle turns (< `gentle_turn_deg`) skip
the spin and stay on the blend.

**Pivot-invariant theta fix (`_apply_theta_correction` override).** The base class
EMA-corrects `remap_theta` after the spin and republishes — but rotating the odom→map
transform about the remap **origin** translates every mapped point by `delta × lever-arm`
(metres for a few degrees at this map's scale). The override re-solves `t` so the robot's
current mapped position is invariant. Safe to re-enable `drift_theta_alpha` now.

### ROUNDABOUT (radial ring + pure-pursuit entry/exit + camera guardrail)

Entered when `map.is_roundabout_node(cur)` or `(next)` **or** `_is_exit_stub(cur)`. Lane is
unreliable here (the solid centre island reads as a short lane segment, the outer line runs
straight out the entrance road), so the **map drives**; the camera only guards the edges.

Three phases, chosen per tick from where the current/next targets sit:

- **Entry approach** (`next` is a ring node but `cur` is not — the window opens one segment
  early, e.g. targeting 29 with next 24): plain pure-pursuit (`_carrot`) on the path, with
  the **path polyline** as the failsafe reference. The ring spline only starts at the ring's
  entry neighbor (`path[first-1]`), so on this segment its nearest point would measure the
  *along-track* distance to that node (~0.7 m) and the off-reference failsafe tripped right
  at the previous node (observed at node 30). The path polyline covers this segment, so it is
  the truthful reference; the spline takes over once `cur` is a ring node (its lead-in covers
  the handover).
- **On the ring** (`cur` and `next` are ring nodes): the carrot rides a per-segment **radial
  arc** through the ring nodes (`radial_ring_curve`, built once and cached, keyed on the ring
  node ids; centre = centroid of the ring nodes). Each ring segment is a polar arc keeping
  each node's own radius — an outlying node → flatter local arc, a tighter node → sharper —
  so the curve never bulges past the node radii (no chord-cutting). Falls back to node
  pure-pursuit when < 3 ring nodes. The nearest-point search along the curve is
  **forward-only** (small backward slack) so odom noise can't snap the carrot to the
  spatially-near exit portion. **The progress index and the failsafe debounce are reset on
  every roundabout window ENTER edge**: a new goal over the same ring reuses the cached
  spline (same ring ids), and without the reset the search stayed parked at the previous
  traversal's tail — every second lap / retry then read a ~0.7 m "excursion" the moment the
  spline branch engaged and emergency-stopped near the ring entry (observed).
- **On the exit approach** (`next` has LEFT the ring, e.g. `28→23→2`): the spline is
  **dropped** for plain pure-pursuit straight at the exit node. The spline's tangent at the
  last ring node points *around* the ring, so riding it made the robot sail past the exit and
  cut the `28→23` edge ~90° off-road. `_carrot` aims into the exit from the start and
  recomputes from the nearest live path node every tick, so it cannot latch onto a stale
  curve endpoint.

`angular.z = base(carrot) + guardrail_nudge`; `linear.x = roundabout_drive_speed`.

**Exit authority (`_is_exit_stub`).** The roundabout window extends through the **exit
stub** — the first non-ring node whose path predecessor is a ring node (e.g. 23 reached from
28) — so the lane controller can't grab the exit drive and follow the ring line off-road. The
lane resumes once the predecessor is no longer a ring node (genuinely out, e.g. at node 2).

**Camera guardrail (`_round_lane_nudge`).** Uses `/lane_controller/info`. NOT lane-following —
returns 0 unless a painted line is *confidently seen* close to the robot centre:
OUTER (road-edge) line too close → nudge **inward**; INNER (island) line too close → nudge
**outward**. Inner/outer is decided per tick from which side the ring centre is on
(CW/CCW-agnostic). Returns 0 on the whole exit approach so the exit drives unopposed.

**Off-reference failsafe.** If the robot strays more than `roundabout_offref_m` from its
**own current reference** (path polyline on the entry approach, spline on the ring, path on
the exit) for `roundabout_offref_ticks` consecutive ticks → EMERGENCY_STOP. Lane is
unreliable here, so this geometric check is the only roundabout safety net. A lenient
lane/map conflict (`c < roundabout_conflict_c`, debounced) also trips it. Heartbeat shows the
active phase: `ROUND src=approach|curve|exit|nodes`.

### FALLBACK (pure-pursuit recovery — now bounded)

Pure-pursuit on the path; recovers to NAVIGATING when lane-OK **and** `dist ≤
recovery_radius` for `recovery_ticks` (inherited hysteresis). FALLBACK runs on a
localization that already proved doubtful (the lane was lost) and used to be the one state
with **no** excursion check — it could drive off-road indefinitely. It now trips the shared
EMERGENCY_STOP on:

- sustained cross-track excursion from the pursued path: `offpath > fallback_offref_m` for
  `fallback_offref_ticks` consecutive ticks;
- timeout: more than `fallback_timeout_s` in FALLBACK (0 = disabled).

### EMERGENCY_STOP (terminal failsafe)

A single terminal halt shared by every failsafe trigger (NAVIGATING conflict, roundabout
conflict / off-reference, FALLBACK off-reference / timeout). On entry it:

- disables the lane controller and brakes actively for ~0.5 s, then goes **silent** on
  `cmd_vel` (a permanent 25 Hz zero stream interleaved with the dashboard's manual/remap
  commands and the robot "struggled to move" — observed);
- **cancels navigation** by publishing an empty goal (waypoint manager → IDLE);
- latches `_emergency` (set/cleared/read under `_lock` — the latch is touched from both the
  control thread and subscriber threads) and publishes `EMERGENCY_STOP` on
  `/orchestrator/state`.

Held at the top of `_step` (above the inactive/DONE handler, so the nav-cancel can't mask
it). **No auto-recovery** — the debounce upstream already absorbs transient noise, so a trip
is a sustained fault. Resume only by issuing a **new goal** from the dashboard (detected in
`_path_cb`). The dashboard shows a pulsing red banner + `⛔ EMERGENCY STOP` badge.

### Object-detection override (TRAFFIC_STOP / STOP_SIGN)

Traffic-light and STOP-sign handling lives in a small, self-contained module —
`object_detection/traffic_sign_handler.py` (`TrafficSignHandler`) — that the orchestrator
calls **via function**, not a state handler. The handler owns a subscriber to
`/object_detection/drive` (JSON detections from the on-robot `perception_node.py`), latches
the light state in its callback, and exposes `evaluate() -> Decision(stop, state)`.

`_step` calls it once per tick, **right after the terminal emergency latch and before the
state snapshot** (so the emergency latch still wins, but a red light/STOP beats all normal
driving):

```
decision = self._obj_det.evaluate()
if decision.stop:
    self._publish_orc_state(decision.state)   # TRAFFIC_STOP / STOP_SIGN
    self._cmd_pub.publish(Twist())            # full stop
    return                                     # self._state untouched
```

- **Red light** → latched `TRAFFIC_STOP`: full stop until a **green** is seen (latched, so a
  frame that detects nothing keeps the last state). Default is GREEN, so a silent/absent
  detection topic never blocks driving.
- **STOP sign** → `STOP_SIGN`: full stop for `stop_sign_hold` seconds (default 3.0), then
  auto-clears and resumes.

Because the override `return`s before touching `self._state`, the FSM is *frozen*, not reset
— `NAVIGATING`/`JUNCTION`/`ROUNDABOUT`/`FALLBACK` all pick up exactly where they were. Set
`traffic_light_enable: false` to ignore detections and run the driving stack on its own.
Pairs with `perception_node.py --mode detection` on the robot (see `architecture.md`).

## Diagnostics — `/orchestrator/diag` (Float64MultiArray, 25 Hz)

The camera-calibration vs odom-drift separator: if `[7]` ≈ 0 (lane says centred) while `[1]`
stays large, the residual is calibration/remap error, not driving error.

| idx | field | idx | field |
|---|---|---|---|
| 0 | state code (0 IDLE, 1 NAV, 2 JUNC, 3 ROUND, 4 FALL, 5 ESTOP, 6 DONE) | 7 | lane `center_offset` (norm) |
| 1 | cross-track to path (m) | 8 | `remap_theta` |
| 2 | `theta_m` (rad) | 9 | `remap_tx` |
| 3 | `theta_l` (rad) | 10 | `remap_ty` |
| 4 | `c` | 11 | dist to target node (m) |
| 5 | `a_lane` | 12 | roundabout off-reference (m) |
| 6 | anti-cut scale | | |

Heartbeat (1 Hz): `st=… node=cur->next lane=… junc=… round=… dist=… off=… hdg_err=… th_l=… c=…`.

## `/lane_controller/info` (Float64MultiArray, published by `lane_controller_node.py`)

Additive — does not touch the existing `cmd_vel`/`state` path. Consumers gate on `len ≥ 8`
(the dashed flags are a later addition, gate on `len ≥ 10`).

```
[0] state_code (0 STOP,1 HOLD,2 TRACKING_CC,3 SINGLE_L,4 SINGLE_R,5 DISABLED)
[1] heading_rad (real lane direction vs robot forward; +ve bends right)
[2] left_valid  [3] right_valid
[4] left_offset [5] right_offset [6] center_offset   (normalized to W/2; |.|~0 = line at robot centre)
[7] lane_width  (normalized to W/2)
[8] left_is_dashed [9] right_is_dashed
```

Dashedness is classified by sampling the class mask (post-`bev_scale`, same coordinates as
the fitted line) in a ±3 px window at 8 rows along the line and voting `lane_dashed` (3) vs
`lane_marking` (2) pixel counts (dashed iff `n_dash ≥ 4 ∧ n_dash > n_solid`).

## Parameters (`orchestrator:` namespace)

Marked **[code]** = in-code default only, not yet in `lane_params.yaml`.

| Param | Default | Meaning |
|---|---|---|
| `use_lane_heading` | `true` | `theta_l` from the real lane heading; `false` = old proxy (A/B switch) |
| `conflict_ticks` | 12 | consecutive `c<0` ticks before EMERGENCY_STOP |
| `conflict_onpath_m` | 0.35 m | only judge a conflict while cross-track to the path is below this |
| `proxy_max_deg` | 80 | full lane steer → this implied heading offset (proxy fallback only) |
| `junction_influence_radius` | 0.50 m | map blend-in distance; must exceed `junction_radius` AND stay below the entry-segment length |
| `junction_turn_full_deg` | 50 | map heading error at which the map fully takes over the approach blend |
| `junction_lane_correct` | True | master enable for the junction anti-cut guardrail |
| `junction_inside_clear` | 0.30 | normalized clearance to the INSIDE line below which the turn is held straighter |
| `junction_lane_gain` | 1.0 | anti-cut damping strength: `scale = 1 - gain*severity` |
| `junction_lane_floor` | 0.0 | minimum turn scale |
| `dash_cross_enable` | `true` | dashed-separator crossing hold |
| `dash_cross_hold_ticks` | 75 | hold budget (~3 s at 25 Hz) |
| `lateral_correct_enable` | True | master enable for the straight-line re-centering |
| `lateral_correct_period` | 50 | run only every N NAVIGATING ticks (~2 s at 25 Hz) |
| `lateral_correct_alpha` | 0.35 | EMA weight per correction of the perpendicular nudge |
| `lateral_centered_clear` | 0.15 | max normalized `center_offset` to count as "centred" |
| `lateral_align_deg` | 15 | max **lane** heading `\|heading_rad\|` to count as "on a straight" (the mapped yaw keeps a fixed 45° sanity bound) |
| `lateral_min_correct_m` | 0.03 | ignore position corrections below this (noise floor) |
| `lateral_max_correct_m` | 0.40 | reject position corrections above this (broken localization) |
| `lateral_theta_alpha` | 0.25 | EMA weight of the straight-line yaw fix (0 = off) |
| `lateral_theta_max_deg` | 10 | reject yaw deltas above this (transient/broken geometry) |
| `roundabout_drive_speed` | 0.10 m/s | map-following speed inside the roundabout window |
| `roundabout_lookahead_m` | 0.25 | carrot lookahead along the ring spline (approach/exit use `lookahead_m` = 0.50) |
| `roundabout_spline_res_m` | 0.03 | sample spacing of the dense ring curve |
| `roundabout_conflict_c` | -0.5 | lenient conflict threshold inside the roundabout |
| `roundabout_lane_correct` | True | master enable for the camera guardrail nudge |
| `roundabout_outer_clear` | 0.25 | normalized clearance to the OUTER line below which we nudge inward |
| `roundabout_inner_clear` | 0.20 | normalized clearance to the INNER line below which we nudge outward |
| `roundabout_lane_gain` | 0.40 | rad/s per unit edge-proximity severity |
| `roundabout_lane_max` | 0.50 | rad/s cap on the guardrail nudge |
| `roundabout_offref_m` | 0.40 m | deviation from the phase reference before EMERGENCY_STOP |
| `roundabout_offref_ticks` | 10 | consecutive off-reference ticks (debounce) |
| `fallback_offref_m` | 0.60 m | FALLBACK cross-track before EMERGENCY_STOP |
| `fallback_offref_ticks` | 25 | debounce (~1 s at 25 Hz) |
| `fallback_timeout_s` | 60 | max time in FALLBACK (0 = no timeout) |
| `offroute_enable` | False | opt-in debounced REPLAN (never seizes steering) |
| `traffic_light_enable` | `true` | object-detection override on; `false` = ignore detections (driving-only) |
| `traffic_light_topic` | `/object_detection/drive` | perception node's JSON detections topic |
| `traffic_light_red_label` | `red_TL` | class name that latches `TRAFFIC_STOP` |
| `traffic_light_green_label` | `green_TL` | class name that releases to GREEN |
| `stop_sign` | `stop_s` | class name that arms a `STOP_SIGN` hold |
| `stop_sign_hold` | 3.0 s | how long to hold at a detected stop sign |

Reused from the base class: `junction_radius`, `junction_align_deg`, `junction_spin_speed`,
`junction_alpha`, `gentle_turn_deg`, `hold_ramp_ticks`, `recovery_ticks`,
`fallback_recovery_radius`, `lookahead_m`, `map_kp`, `map_max_angular`, drift params.

## Running it

```bash
# normally: started by start_all.sh (ORCHESTRATOR_NAME="n_orchestrator.py")
cd jetauto_autonomous && ./start_all.sh

# or standalone (roscore + params already up)
python2 scripts/n_orchestrator.py
```

The base `Orchestrator` class is defined inline in the same file — the only external imports
are `map_loader` and `object_detection.traffic_sign_handler` (the traffic-light/STOP handler).

## Offline simulation

`decide_blend(...)` and `radial_ring_curve(...)` are pure functions (no ROS). The full node
runs against a ROS stub in `testing/sim/`:

```bash
python3 testing/sim/run_sim.py --scenario all --orc nn
```

7 scenarios: node-6 left turn, roundabout exit 28→23→2, 0.5°/s yaw drift, +5° camera bias,
wrong-way conflict stop, lane-dropout→FALLBACK→recovery, FALLBACK runaway stop.

## On-robot checklist

1. Watch `straight correction:` lines — `dtheta` small and steady (a few tenths of a degree
   per fire), `theta` slowly tracking; `xy=0` fires are yaw-only re-anchors.
2. `/orchestrator/diag`: record a lap; plot `[1]` vs `[7]` — persistent `[7]`≉0 while
   centred indicates BEV/camera miscalibration (measurable now).
3. Roundabout: heartbeat `ROUND src=` must walk `approach → curve → exit`; an off-reference
   stop right at the ring entry on a *second* goal would mean the progress-index reset
   regressed.
4. If behaviour regresses: `use_lane_heading: false` (proxy A/B) and/or
   `lateral_theta_alpha: 0.0` (yaw fix off) — each fix is independently disableable.