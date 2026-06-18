# on_jetauto_scripts — utility scripts for the JetAuto

One-off and maintenance scripts that run on the robot or on a development machine.
None of these are part of the live driving stack; they support setup, debugging,
dataset collection, and model conversion.

> **Note:** The on-robot perception node (`perception_node.py`) has moved to
> `jetauto_autonomous/perception/` and is launched via `jetauto_autonomous/run-models.sh`.
> The former `drive_segm/lane_follower.py` is superseded and kept here for reference only.

---

## Subfolders

### `drive_segm/`

The original segmentation-only node and its BEV helper — superseded on-robot by
`jetauto_autonomous/perception/perception_node.py`.

| File | Purpose |
|---|---|
| `lane_follower.py` | Standalone lane-segmentation ROS node (Python 3, ONNX/TensorRT). Publishes `/lane_mask_bev` only. Superseded on-robot by `perception_node.py --mode segmentation`. Kept as a reference and for machines without the YOLO11 engine. |
| `bev_from_config.py` | Legacy BEV helper (config-file-driven warp). Not used by the current stack; kept for reference. |

---

### `tensor_rt/`

TensorRT conversion utilities.

| File | Purpose |
|---|---|
| `convert_to_tensor_rt.py` | Converts an ONNX model to a TensorRT `.engine` file on the Jetson. The resulting engine is placed in `jetauto_autonomous/perception/models/`. |

---

### `sh/`

Robot maintenance shell scripts.

| File | Purpose |
|---|---|
| `master-node-restart.sh` | Restarts the ROS master node (roscore). Use when roscore hangs without a full reboot. |
| `record-camera.sh` | Records the camera topic (`/depth_cam/rgb/image_raw`) to a ROS bag for offline replay and dataset extraction. |
| `restart-ros.sh` | Full ROS environment restart (kills all ROS processes and restarts roscore). |

---

### `extrac_frames/`

Dataset-collection utilities.

| File | Purpose |
|---|---|
| `extract-frames-directly.py` | Extracts JPEG frames from a recorded video or ROS bag at a configurable rate. Used to build the LabelMe annotation dataset for training. |

---

### `mapping/`

Map-building utilities.

| File | Purpose |
|---|---|
| `map-auto.py` | Assisted waypoint-map builder: drive the robot manually while the script records odom poses, then outputs a YAML map file consumable by `waypoint_manager_node.py`. |

---

### `testing/`

Low-level model and engine smoke tests (no ROS needed).

| File | Purpose |
|---|---|
| `test-onnx.py` | Loads a segmentation `.onnx` model and runs a single forward pass on a synthetic input to verify the export is valid. |
| `test-engine.py` | Loads a TensorRT `.engine` file and runs a single forward pass; checks output shape and timing. Used to validate `convert_to_tensor_rt.py` output before deploying to the robot. |
