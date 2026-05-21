# Baris Vision Data Capture

GUI dataset capture tool for `/latest_serving_decision` events.

Run:

```bash
./run_dataset_capture_gui.sh
```

The launcher sources `/opt/ros/humble/setup.bash` before starting the GUI.

Default output root:

```text
/home/xyz-ai/baris_vision_data_capture/datasets
```

Naming:

```text
<site_code>_<timestamp>_<camera_kind>_<type>/
  images/
    <site_code>_<timestamp+timezone>_<camera_kind>_<type>_frame_00000000.png
  bbox/
    <site_code>_<timestamp>_<camera_kind>_<type>_frame_00000000.json
  metadata.json
  frame_manifest.jsonl
```

The tool does not generate duplicate review artifacts such as `yolo_results.jsonl`,
`color.mp4`, or `color_yolo_bbox.mp4`. YOLO is used only to produce LabelMe JSON
files under `bbox/`.

Frames are saved at the source image resolution. The observed width and height are
recorded in `metadata.json` and each LabelMe JSON file.

Example:

```text
BBXYZ_20260519T131820_topview1_serving/
  images/BBXYZ_20260519T131820+0900_topview1_serving_frame_00000000.png
  bbox/BBXYZ_20260519T131820_topview1_serving_frame_00000000.json
```

CLI mode is also available:

```bash
./run_dataset_capture_gui.sh --no-gui --site-code BBXYZ --camera-kind topview1 --dataset-type serving --max-events 3
```
