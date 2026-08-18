# UniDrive — Autonomous Driving on JetAuto

An autonomous-driving stack for the Hiwonder **JetAuto** robot
(Jetson Nano, ROS1 Melodic). A semantic-segmentation network detects lane markings from the
onboard camera; a Hough-based lane controller, a Dijkstra waypoint planner and a
disagreement-driven orchestrator steer the robot along a pre-built map of the track —
handling junctions, a roundabout, dashed-separator lane changes, on-line odometry drift
correction and terminal safety stops.

```
camera ─► perception_node.py ──► /lane_mask_bev ─► lane_controller ─► proposed cmd_vel
 (640×480)  (Py3, ONNX/TensorRT,    (BEV mask)      (Py2.7, Hough)        │
             detection + segm.)  └► /object_detection/drive ──────────────►│
                                     (JSON detections)                      ▼
            waypoint_manager (Py2.7, Dijkstra) ─► nav_info ─►  n_orchestrator (Py2.7)
                                                               THE ONLY NODE THAT DRIVES
                                                                         │
                                                               /jetauto_controller/cmd_vel
```

Two separate Python environments communicate only via ROS topics: the neural network runs
in **Python 3** (conda, ONNX Runtime or TensorRT), all control nodes run in **Python 2.7**
(system ROS Melodic). No catkin workspace — plain scripts.

## Repository layout

| Path | Content |
|---|---|
| `jetauto_autonomous/` | The on-robot stack: perception node, lane controller, waypoint manager, orchestrator, web dashboard, track map, `start_all.sh`/`stop_all.sh`, `run-models.sh`/`stop-models.sh` |
| `jetauto_autonomous/perception/` | `perception_node.py` (Python 3) — merged detection + segmentation node + `auto_calibration.py` + `bev_calibration_session.py` + TensorRT engines (`models/`) |
| `jetauto_autonomous/docs/` | [Architecture](jetauto_autonomous/docs/architecture.md), [setup/usage README](jetauto_autonomous/docs/README.md), ROS topic reference, LaTeX report |
| `on_jetauto_scripts/` | Utility scripts for the robot: maintenance shells, TensorRT conversion, dataset extraction, map builder, model smoke tests — see [`on_jetauto_scripts/README.md`](on_jetauto_scripts/README.md) |
| `on_jetauto_scripts/drive_segm/` | Superseded segmentation-only node `lane_follower.py` — kept for reference |
| `pipeline/` | Model training: LabelMe → dataset → MobileNetV3/SegFormer training → evaluation → ONNX export |
| `testing/` | `offline_tester.py` (run the full vision/control pipeline on a video, no ROS) and `sim/` (closed-loop orchestrator simulation against a ROS stub) |
| [`jetauto_autonomous/docs/internal_docs/n_orchestrator.md`](jetauto_autonomous/docs/internal_docs/n_orchestrator.md) | Detailed design doc of the orchestrator (blend, roundabout, failsafes, parameters) |

## Quick start (on the robot)

```bash
# 1. Control stack (Python 2.7 — roscore must already be running)
cd jetauto_autonomous
./start_all.sh            # rosbridge, video server, dashboard, controller, planner, orchestrator

# 2. Perception node — object detection + lane segmentation (Python 3 conda env)
cd jetauto_autonomous
./run-models.sh           # both models (default); engines in perception/models/
# ./run-models.sh detection     # detection-only (no lane segmentation)
# ./run-models.sh segmentation  # segmentation-only (no object detection)

# 3. Open the dashboard and send a goal
#    http://<ROBOT_IP>:8000/
```

The perception node can also be started, stopped and **calibrated from the dashboard**
(the "Percezione" bar and the 📐 Calibra BEV panel) — set `perception/conda_env` in
`config/lane_params.yaml` first so `run-models.sh` can activate the right conda
environment on its own. Without a `perception/calibration.json` the dashboard offers a
calibration as soon as the models are up. Stop the perception node with
`./stop-models.sh`; stop everything else with `./stop_all.sh`.

See [`jetauto_autonomous/docs/README.md`](jetauto_autonomous/docs/README.md) for
prerequisites, installation, tuning and troubleshooting.

## Training a model

```bash
cd pipeline                       # edit config.yaml first
python3 1_prepare_dataset.py --config config.yaml --preview 10
python3 3_train.py --config config.yaml          # default MobileNetV3+LR-ASPP; --model segformer-b0|segformer-b1|fastscnn to switch
python3 4_evaluate.py --checkpoint checkpoints/best.pth
python3 5_export.py --checkpoint checkpoints/best.pth --simplify --verify
# on the Jetson:
trtexec --onnx=exports/model.onnx --saveEngine=exports/model.trt --fp16
```

Classes: `0=background, 1=road, 2=lane_marking, 3=lane_dashed, 4=zebra`; input 320×128 from
the bottom 55% of the camera frame. See [`pipeline/README.md`](pipeline/README.md).

## Highlights

- **Single-driver architecture** — exactly one node publishes the hardware command topic;
  lane controller and planner only *propose*.
- **Disagreement-driven lane/map blend** — the lane drives on open road; the map takes over
  proportionally to how hard it wants to turn and how much the lane disagrees.
- **Map-aware roundabout** — radial-arc reference curve through the ring nodes, camera-frame
  edge guardrail, pure-pursuit entry/exit handling.
- **On-line localization correction** — the camera pins the map frame (lateral + yaw) while
  confidently centred on straights; junction alignments provide yaw fixes that pivot the
  frame about the robot.
- **Bounded failure modes** — every failsafe (wrong-way conflict, off-reference excursions,
  fallback timeout) converges on a terminal, debounced `EMERGENCY_STOP`.
- **Testable without hardware** — pure-logic core modules, an offline video tester, and a
  deterministic closed-loop simulator that re-runs the historical failure scenarios.
- **Reversible BEV calibration** — the warp is recalibrated from the web UI with a visual
  preview of the picked lane corners; the candidate is staged in the perception node and
  the live warp is only touched on apply, so aborting cannot break a working setup.

## Platform

Jetson Nano (JetPack 4.6 / L4T 32.7), Ubuntu 18.04, ROS Melodic, CUDA 10.2, TensorRT 8.2,
OpenCV 4.5 with CUDA. Model: MobileNetV3-Large + LR-ASPP (default; SegFormer B0/B1 optional)
exported to ONNX/TensorRT, 20–30 FPS.

## License

[Apache 2.0](LICENSE)