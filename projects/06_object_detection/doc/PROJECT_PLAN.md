# Project 6: Object Detection & Tracking (Hard)

## Goal

Build a **production-grade object detection and tracking system** that detects objects
in images and tracks them across video frames in real-time. The system uses YOLOv8 for
detection and DeepSORT for multi-object tracking, optimized with TensorRT for edge
deployment. By the end, you will understand how computer vision pipelines work in
autonomous vehicles, surveillance systems, and retail analytics -- from model fine-tuning
to real-time WebSocket streaming and edge device deployment.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why object detection? | Detection is the foundation of most real-world computer vision: autonomous driving, robotics, medical imaging, retail analytics, security. It is the most in-demand CV skill. |
| Why YOLO? | YOLOv8 (Ultralytics) is the current industry standard for real-time detection. It is fast, accurate, and has excellent tooling. Learning YOLO is directly applicable to industry jobs. |
| Why add tracking? | Detection on single frames is useful, but tracking objects across video frames is what production systems actually need. DeepSORT adds identity persistence across frames. |
| How is this different from classification? | Classification says "this image contains a cat." Detection says "there is a cat at coordinates (x1,y1,x2,y2) with 94% confidence, and a dog at (x3,y3,x4,y4) with 87% confidence." Much harder. |
| What new skills will I learn? | Bounding box regression, anchor-based detection, video processing with OpenCV, real-time inference optimization (TensorRT/ONNX), WebSocket streaming, and edge deployment. |

---

## Architecture Overview

```
                    ┌──────────────────┐
                    │   Video Source   │
                    │ (Camera/File/    │
                    │  RTSP Stream)    │
                    └────────┬─────────┘
                             │ Frames
                             ▼
                    ┌──────────────────┐
                    │   FastAPI +      │
                    │   WebSocket      │
                    │   Server         │
                    └───┬────┬────┬────┘
                        │    │    │
           ┌────────────┘    │    └────────────┐
           ▼                 ▼                  ▼
    ┌────────────┐   ┌────────────────┐  ┌────────────┐
    │  YOLOv8    │   │   DeepSORT     │  │  TensorRT  │
    │  Detector  │──▶│   Tracker      │  │  Optimizer │
    │(Ultralytics)│   │ (ID Tracking)  │  │ (INT8/FP16)│
    └────────────┘   └───────┬────────┘  └────────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │  MLflow  │ │Prometheus│ │ Results  │
         │ Registry │ │ Metrics  │ │ Storage  │
         └──────────┘ └────┬─────┘ └──────────┘
                           ▼
                     ┌──────────┐
                     │ Grafana  │
                     │Dashboard │
                     └──────────┘

Edge deployment: TensorRT-optimized model on Jetson Nano / Raspberry Pi.
```

---

## Tech Stack

| Category | Tool | Why This One |
|----------|------|-------------|
| Language | Python 3.11+ | CV ecosystem standard |
| Detection Framework | Ultralytics YOLOv8 | State-of-the-art real-time detection, excellent API |
| Tracking | DeepSORT | Proven multi-object tracker, combines detection + ReID |
| Video Processing | OpenCV | Industry standard for video I/O and frame manipulation |
| Model Optimization | TensorRT / ONNX Runtime | 2-5x inference speedup, essential for real-time |
| API Framework | FastAPI + WebSocket | REST for images, WebSocket for video streams |
| Experiment Tracking | MLflow | Track detection experiments and model versions |
| Annotation Tool | CVAT or Label Studio | For custom dataset annotation |
| Containerization | Docker + docker-compose | Orchestrate detection services |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Monitoring | Prometheus + Grafana | Track FPS, detection quality, system resources |
| Edge Platform | NVIDIA Jetson Nano or Raspberry Pi | Edge deployment target |
| Testing | pytest | Unit and integration testing |

---

## Project Structure

```
object-detection-tracking/
│
├── doc/
│   ├── DESIGN_DOC.md                # Problem statement, constraints, success criteria
│   ├── MODEL_CARD.md                # Model documentation
│   └── PROJECT_PLAN.md              # This file
│
├── pyproject.toml                   # Dependencies and project metadata
├── dvc.yaml                         # Data pipeline definition
│
├── configs/
│   ├── train_config.yaml            # YOLOv8 training hyperparameters
│   ├── serve_config.yaml            # Serving configuration
│   ├── data_config.yaml             # Dataset paths, augmentation settings
│   └── edge_config.yaml             # Edge deployment optimization settings
│
├── data/                            # Git-ignored, DVC-tracked
│   ├── raw/                         # Original COCO subset
│   │   ├── images/
│   │   │   ├── train2017/
│   │   │   └── val2017/
│   │   └── annotations/
│   │       ├── instances_train2017.json
│   │       └── instances_val2017.json
│   ├── processed/                   # YOLO-format annotations
│   │   ├── images/
│   │   └── labels/
│   ├── augmented/                   # Mosaic/mixup augmented data
│   └── custom/                      # Custom annotated dataset (optional)
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Dataset exploration, class distribution
│   ├── 02_augmentation_viz.ipynb    # Visualize augmentation effects
│   ├── 03_training_analysis.ipynb   # Training curves, loss components
│   ├── 04_evaluation.ipynb          # mAP analysis, per-class performance
│   └── 05_speed_benchmarks.ipynb    # Latency comparison across runtimes
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py              # Download COCO subset
│   │   ├── convert.py               # COCO JSON -> YOLO format conversion
│   │   ├── augment.py               # Mosaic, mixup, and detection-specific augmentation
│   │   ├── dataset.py               # Custom dataset class for fine-tuning
│   │   └── validate.py              # Annotation integrity checks
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── detector.py              # YOLOv8 fine-tuning wrapper
│   │   ├── tracker.py               # DeepSORT tracking integration
│   │   ├── export.py                # Export to ONNX / TensorRT
│   │   └── optimize.py              # Quantization, pruning
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                 # YOLOv8 training script
│   │   ├── evaluate.py              # mAP evaluation, per-class analysis
│   │   ├── callbacks.py             # Custom training callbacks
│   │   └── anchor_optim.py          # Anchor box optimization for custom data
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI application
│   │   ├── schemas.py               # Request/response Pydantic models
│   │   ├── detect.py                # Single image detection logic
│   │   ├── stream.py                # WebSocket video stream processing
│   │   ├── batch.py                 # Batch image processing endpoint
│   │   └── visualize.py             # Draw bounding boxes, tracks on frames
│   │
│   ├── edge/
│   │   ├── __init__.py
│   │   ├── tensorrt_engine.py       # TensorRT inference engine
│   │   ├── onnx_runtime.py          # ONNX Runtime inference
│   │   ├── quantize.py              # INT8/FP16 quantization
│   │   └── benchmark.py             # Speed/accuracy benchmarking
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Prometheus metric definitions
│   │   ├── fps_tracker.py           # Real-time FPS monitoring
│   │   └── class_distribution.py    # Detected class distribution over time
│   │
│   └── frontend/
│       └── app.py                   # Streamlit UI for image/video upload
│
├── tests/
│   ├── unit/
│   │   ├── test_convert.py          # Annotation format conversion
│   │   ├── test_augment.py          # Augmentation output validity
│   │   ├── test_detector.py         # Detection output shape, NMS
│   │   ├── test_tracker.py          # Track ID assignment
│   │   ├── test_schemas.py          # API schema validation
│   │   └── test_visualize.py        # Bounding box drawing
│   ├── integration/
│   │   ├── test_api.py              # Full detection request pipeline
│   │   ├── test_stream.py           # WebSocket stream test
│   │   ├── test_batch.py            # Batch processing test
│   │   └── test_training.py         # Training runs one epoch
│   └── conftest.py                  # Shared fixtures (sample images, video clips)
│
├── docker/
│   ├── Dockerfile.api               # Detection API container
│   ├── Dockerfile.frontend          # Streamlit container
│   ├── Dockerfile.training          # Training with GPU support
│   └── Dockerfile.edge              # Minimal container for edge deployment
│
├── docker-compose.yaml              # API + Frontend + Monitoring
│
├── .github/
│   └── workflows/
│       ├── ci.yaml                  # Lint -> Test -> Build on PR
│       └── cd.yaml                  # Deploy on merge to main
│
├── grafana/
│   └── dashboards/
│       └── detection_monitoring.json # Detection quality dashboard
│
├── prometheus/
│   └── prometheus.yml               # Scrape config
│
├── edge/
│   ├── jetson_setup.sh              # Jetson Nano setup script
│   ├── rpi_setup.sh                 # Raspberry Pi setup script
│   └── run_edge.py                  # Edge inference script
│
└── scripts/
    ├── setup.sh                     # One-command project setup
    ├── train.sh                     # Run training with default config
    ├── benchmark.sh                 # Run speed benchmarks
    ├── export_tensorrt.sh           # Export model to TensorRT
    └── annotate.sh                  # Launch annotation tool
```

---

## Phase 1: Project Setup & Design Doc

**Duration:** 1-2 days
**Objective:** Define the detection and tracking problem, set up the development environment.

### Tasks

1. **Write `DESIGN_DOC.md`**
   - **Problem statement:** "Given an image or video stream, detect all objects of
     specified classes, draw bounding boxes with confidence scores, and track object
     identities across video frames."
   - **Success criteria:**
     - mAP@50 >= 0.65 on COCO val set (fine-tuned subset)
     - mAP@50:95 >= 0.45
     - Inference speed >= 30 FPS on GPU (RTX 3060 or equivalent)
     - Inference speed >= 15 FPS with TensorRT on Jetson Nano
     - Tracking MOTA >= 0.50 on test video sequences
     - API latency < 100ms per frame on GPU
   - **Target classes:** Start with a subset of COCO (person, car, bicycle, dog, cat --
     5-10 classes). This keeps training manageable while learning the full pipeline.
   - **Out of scope:** instance segmentation, 3D detection, pose estimation, action recognition
   - **Risks:** GPU memory constraints during training, real-time processing bottlenecks,
     annotation quality for custom data, edge device compatibility

2. **Initialize the repository**
   - `git init`, create `.gitignore` (data/, models/, runs/, *.engine, *.onnx, etc.)
   - Create `pyproject.toml`:
     ```toml
     [project]
     name = "object-detection-tracking"
     version = "0.1.0"
     requires-python = ">=3.11"
     dependencies = [
         "ultralytics>=8.0",
         "opencv-python>=4.8",
         "deep-sort-realtime>=1.3",
         "fastapi>=0.100",
         "uvicorn>=0.23",
         "websockets>=12.0",
         "onnxruntime-gpu>=1.16",
         "mlflow>=2.8",
         "streamlit>=1.28",
         "prometheus-client>=0.19",
         "pydantic>=2.0",
         "Pillow>=10.0",
         "numpy>=1.24",
     ]
     ```

3. **Create the folder structure** (as shown above)

4. **Set up development environment**
   - Install CUDA toolkit and cuDNN (required for GPU training and TensorRT)
   - Verify GPU detection: `torch.cuda.is_available()`
   - Install Ultralytics: `pip install ultralytics` and verify with `yolo predict`
   - Install TensorRT (optional for Phase 6, but verify compatibility now)

### Skills Learned

- Understanding detection vs classification problem formulation
- Setting up GPU-accelerated development environments
- CUDA/cuDNN compatibility management

---

## Phase 2: Data Pipeline

**Duration:** 4-5 days
**Objective:** Download and prepare COCO data in YOLO format, with detection-specific augmentation.

### Tasks

1. **Download COCO Subset** -- `src/data/download.py`
   - Download COCO 2017 train and val images + annotations
   - Full COCO is 20GB+ images. Start with a subset: filter to your target classes only
   - Download script should be resumable (large files)
   ```python
   COCO_URLS = {
       "train_images": "http://images.cocodataset.org/zips/train2017.zip",
       "val_images": "http://images.cocodataset.org/zips/val2017.zip",
       "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
   }

   def filter_coco_classes(
       annotation_file: Path,
       target_classes: list[str],
   ) -> dict:
       """Filter COCO annotations to keep only target classes."""
       with open(annotation_file) as f:
           coco = json.load(f)
       target_cat_ids = [
           c["id"] for c in coco["categories"]
           if c["name"] in target_classes
       ]
       filtered_annotations = [
           a for a in coco["annotations"]
           if a["category_id"] in target_cat_ids
       ]
       # Keep only images that have at least one target annotation
       image_ids = {a["image_id"] for a in filtered_annotations}
       filtered_images = [i for i in coco["images"] if i["id"] in image_ids]
       logger.info(f"Filtered: {len(filtered_images)} images, "
                   f"{len(filtered_annotations)} annotations")
       return {
           "images": filtered_images,
           "annotations": filtered_annotations,
           "categories": [c for c in coco["categories"] if c["id"] in target_cat_ids],
       }
   ```

2. **Annotation Format Conversion** -- `src/data/convert.py`
   - COCO uses JSON with absolute pixel coordinates. YOLO uses text files with
     normalized coordinates (0-1).
   - Convert COCO format to YOLO format:
   ```python
   def coco_to_yolo(coco_annotation: dict, image_width: int, image_height: int) -> str:
       """Convert a single COCO annotation to YOLO format.

       COCO: [x_min, y_min, width, height] in pixels
       YOLO: [class_id, x_center, y_center, width, height] normalized 0-1
       """
       x_min, y_min, w, h = coco_annotation["bbox"]
       x_center = (x_min + w / 2) / image_width
       y_center = (y_min + h / 2) / image_height
       w_norm = w / image_width
       h_norm = h / image_height
       class_id = class_mapping[coco_annotation["category_id"]]
       return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
   ```
   - Validate conversions: visualize a few images with both COCO and YOLO annotations
     to confirm they match

3. **Exploratory Data Analysis** -- `notebooks/01_eda.ipynb`
   - **Class distribution:** how many annotations per class? Detection datasets are
     often highly imbalanced (many more "person" than "bicycle" in COCO)
   - **Bounding box size distribution:** histogram of box areas (small/medium/large).
     Small objects are much harder to detect.
     ```python
     areas = [a["area"] for a in annotations]
     small = sum(1 for a in areas if a < 32**2)
     medium = sum(1 for a in areas if 32**2 <= a < 96**2)
     large = sum(1 for a in areas if a >= 96**2)
     print(f"Small: {small}, Medium: {medium}, Large: {large}")
     # COCO defines: small < 32^2, medium 32^2 - 96^2, large > 96^2
     ```
   - **Aspect ratio distribution:** most boxes are roughly square, but some classes
     (e.g., person) tend to be tall and narrow
   - **Objects per image:** histogram. Some images have 1 object, others have 50+.
   - **Spatial distribution:** heatmap of where objects tend to appear in images.
     Center bias is common.
   - **Sample visualizations:** display 16 images with ground truth bounding boxes
     and class labels

4. **Detection-Specific Data Augmentation** -- `src/data/augment.py`
   - Detection augmentation is fundamentally different from classification augmentation.
     You must transform both the image AND the bounding boxes.
   - **Mosaic augmentation:** combine 4 training images into one, creating a mosaic.
     This exposes the model to more objects per image and more context.
     ```python
     def mosaic_augmentation(
         images: list[np.ndarray],
         labels: list[np.ndarray],
         target_size: int = 640,
     ) -> tuple[np.ndarray, np.ndarray]:
         """Create a mosaic from 4 images with their bounding boxes."""
         mosaic = np.zeros((target_size, target_size, 3), dtype=np.uint8)
         # Random center point for the mosaic split
         cx = random.randint(target_size // 4, 3 * target_size // 4)
         cy = random.randint(target_size // 4, 3 * target_size // 4)
         # Place 4 images in quadrants, resize and adjust bounding boxes
         placements = [
             (0, 0, cx, cy),           # top-left
             (cx, 0, target_size, cy),  # top-right
             (0, cy, cx, target_size),  # bottom-left
             (cx, cy, target_size, target_size),  # bottom-right
         ]
         all_labels = []
         for img, lbls, (x1, y1, x2, y2) in zip(images, labels, placements):
             h, w = y2 - y1, x2 - x1
             resized = cv2.resize(img, (w, h))
             mosaic[y1:y2, x1:x2] = resized
             # Adjust label coordinates to mosaic space
             adjusted = adjust_boxes(lbls, img.shape, (x1, y1, x2, y2), target_size)
             all_labels.append(adjusted)
         return mosaic, np.concatenate(all_labels)
     ```
   - **Mixup augmentation:** blend two images and their labels together
   - **Standard augmentations that work with boxes:**
     - Random horizontal flip (flip boxes too)
     - Random scale (0.5x - 1.5x)
     - Color jitter (does not affect boxes)
     - Random crop (must filter out boxes that are mostly cropped away)
   - **Important:** after augmentation, validate that all bounding boxes are still
     valid (within image bounds, non-zero area)

5. **Augmentation Visualization** -- `notebooks/02_augmentation_viz.ipynb`
   - Visualize each augmentation type on sample images
   - Show before/after with bounding boxes
   - Verify boxes are correctly transformed

6. **Custom Dataset Class** -- `src/data/dataset.py`
   - While Ultralytics handles its own data loading, write a custom dataset class
     for flexibility and understanding:
   ```python
   class DetectionDataset(Dataset):
       """YOLO-format detection dataset."""
       def __init__(self, image_dir: Path, label_dir: Path, transform=None):
           self.image_paths = sorted(image_dir.glob("*.jpg"))
           self.label_dir = label_dir
           self.transform = transform

       def __getitem__(self, idx):
           image = cv2.imread(str(self.image_paths[idx]))
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           label_path = self.label_dir / self.image_paths[idx].stem + ".txt"
           labels = self._parse_yolo_labels(label_path)
           if self.transform:
               image, labels = self.transform(image, labels)
           return image, labels

       def _parse_yolo_labels(self, path: Path) -> np.ndarray:
           """Parse YOLO format: class_id x_center y_center width height."""
           if not path.exists():
               return np.zeros((0, 5))
           with open(path) as f:
               labels = [list(map(float, line.split())) for line in f.readlines()]
           return np.array(labels)
   ```

7. **Annotation Validation** -- `src/data/validate.py`
   - Check for: missing label files, empty label files, invalid coordinates,
     overlapping annotations, unreasonably small boxes
   - Generate validation report with statistics

8. **Define DVC pipeline** -- `dvc.yaml`
   - Stage 1: download -> Stage 2: filter classes -> Stage 3: convert format ->
     Stage 4: validate -> Stage 5: augment

### Skills Learned

- Working with COCO dataset format and annotations
- Annotation format conversion (COCO to YOLO)
- Detection-specific EDA (box sizes, aspect ratios, spatial distribution)
- Detection-aware data augmentation (mosaic, mixup with box transforms)
- Understanding how augmentation differs for detection vs classification

---

## Phase 3: Model Development

**Duration:** 5-7 days
**Objective:** Fine-tune YOLOv8 on the target classes, optimize anchors, and add multi-object tracking.

### Tasks

1. **YOLOv8 Fine-Tuning** -- `src/model/detector.py`
   - Start with a YOLOv8s (small) pretrained on COCO. Fine-tune on your class subset.
   - Ultralytics makes this straightforward:
   ```python
   from ultralytics import YOLO

   def train_yolov8(
       data_yaml: str,
       model_size: str = "yolov8s.pt",
       epochs: int = 100,
       imgsz: int = 640,
       batch_size: int = 16,
   ) -> YOLO:
       """Fine-tune YOLOv8 on custom dataset."""
       model = YOLO(model_size)
       results = model.train(
           data=data_yaml,
           epochs=epochs,
           imgsz=imgsz,
           batch=batch_size,
           patience=20,           # Early stopping patience
           save=True,
           save_period=10,        # Save checkpoint every 10 epochs
           plots=True,            # Generate training plots
           mosaic=1.0,            # Enable mosaic augmentation
           mixup=0.1,             # Enable mixup augmentation
           degrees=10.0,          # Random rotation
           scale=0.5,             # Random scale
           fliplr=0.5,            # Horizontal flip probability
           device="0",            # GPU device
       )
       return model
   ```
   - Create the YOLO data configuration file:
   ```yaml
   # data.yaml
   path: ./data/processed
   train: images/train
   val: images/val
   test: images/test

   nc: 5  # number of classes
   names: ["person", "car", "bicycle", "dog", "cat"]
   ```

2. **Training Script** -- `src/training/train.py`
   - Wrapper around Ultralytics training with MLflow integration
   - Log all hyperparameters, training curves, and final metrics to MLflow
   ```python
   def train_with_tracking(config: TrainConfig) -> None:
       """Train YOLOv8 with MLflow experiment tracking."""
       with mlflow.start_run(run_name=config.experiment_name):
           mlflow.log_params({
               "model_size": config.model_size,
               "epochs": config.epochs,
               "batch_size": config.batch_size,
               "imgsz": config.imgsz,
               "learning_rate": config.lr0,
               "mosaic": config.mosaic,
               "n_classes": config.nc,
           })

           model = YOLO(config.model_size)
           results = model.train(
               data=config.data_yaml,
               epochs=config.epochs,
               imgsz=config.imgsz,
               batch=config.batch_size,
               lr0=config.lr0,
           )

           # Log final metrics
           metrics = model.val()
           mlflow.log_metrics({
               "mAP50": metrics.box.map50,
               "mAP50_95": metrics.box.map,
               "precision": metrics.box.mp,
               "recall": metrics.box.mr,
           })

           # Log per-class AP
           for i, class_name in enumerate(config.class_names):
               mlflow.log_metric(f"AP50_{class_name}", metrics.box.ap50[i])

           # Log model artifact
           mlflow.log_artifact(str(model.trainer.best))
   ```

3. **Anchor Box Optimization** -- `src/training/anchor_optim.py`
   - YOLOv8 uses anchor-free detection, but understanding anchors is still valuable
     for YOLOv5 comparisons and custom architectures
   - Analyze your dataset's bounding box dimensions and verify they align with
     the model's detection range
   ```python
   def analyze_box_statistics(label_dir: Path) -> dict:
       """Analyze bounding box dimensions for anchor optimization."""
       widths, heights = [], []
       for label_file in label_dir.glob("*.txt"):
           with open(label_file) as f:
               for line in f:
                   parts = line.strip().split()
                   widths.append(float(parts[3]))   # normalized width
                   heights.append(float(parts[4]))   # normalized height
       return {
           "mean_width": np.mean(widths),
           "mean_height": np.mean(heights),
           "median_width": np.median(widths),
           "median_height": np.median(heights),
           "aspect_ratios": [w/h for w, h in zip(widths, heights)],
       }
   ```

4. **Multi-Scale Training**
   - Train with multiple input resolutions (480, 640, 800) to make the model
     robust to different image sizes
   - Ultralytics supports this natively with the `imgsz` parameter during training
   - Experiment: does multi-scale training improve small object detection?

5. **DeepSORT Integration** -- `src/model/tracker.py`
   - DeepSORT adds persistent identity tracking across video frames
   - It combines: YOLO detections + Kalman filter (motion prediction) +
     appearance features (Re-ID network)
   ```python
   from deep_sort_realtime.deepsort_tracker import DeepSort

   class ObjectTracker:
       def __init__(
           self,
           detector: YOLO,
           max_age: int = 30,          # Frames to keep lost track
           n_init: int = 3,            # Detections before confirmed track
           max_cosine_distance: float = 0.3,
       ):
           self.detector = detector
           self.tracker = DeepSort(
               max_age=max_age,
               n_init=n_init,
               max_cosine_distance=max_cosine_distance,
           )

       def process_frame(self, frame: np.ndarray) -> list[TrackedObject]:
           """Detect objects and update tracks for a single frame."""
           # Run YOLOv8 detection
           results = self.detector(frame, verbose=False)[0]
           detections = []
           for box in results.boxes:
               x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
               confidence = box.conf[0].cpu().item()
               class_id = int(box.cls[0].cpu().item())
               detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

           # Update tracker
           tracks = self.tracker.update_tracks(detections, frame=frame)
           tracked_objects = []
           for track in tracks:
               if not track.is_confirmed():
                   continue
               tracked_objects.append(TrackedObject(
                   track_id=track.track_id,
                   bbox=track.to_ltrb(),     # [left, top, right, bottom]
                   class_id=track.det_class,
                   confidence=track.det_conf,
               ))
           return tracked_objects

       def process_video(self, video_path: str) -> Generator[list[TrackedObject], None, None]:
           """Process video frame by frame, yielding tracked objects."""
           cap = cv2.VideoCapture(video_path)
           while cap.isOpened():
               ret, frame = cap.read()
               if not ret:
                   break
               yield self.process_frame(frame)
           cap.release()
   ```

6. **Run Experiments** (track all in MLflow)

   | Experiment | Model | Key Change | Expected Result |
   |-----------|-------|-----------|----------------|
   | Baseline | YOLOv8s (pretrained) | No fine-tuning, eval on subset | mAP50 ~0.60 (already trained on COCO) |
   | Fine-tune-50ep | YOLOv8s | 50 epochs fine-tuning | mAP50 ~0.65 |
   | Fine-tune-100ep | YOLOv8s | 100 epochs | mAP50 ~0.68, check overfitting |
   | Multi-scale | YOLOv8s | Multi-scale training | Better small object AP |
   | YOLOv8m | YOLOv8m (medium) | Larger model | mAP50 ~0.72, slower |
   | Strong-aug | YOLOv8s | Mosaic + mixup + heavy aug | Better generalization |

7. **Pick the best model** using MLflow UI
   - Compare mAP vs inference speed tradeoff
   - For real-time use: YOLOv8s (fast, good accuracy)
   - For accuracy-critical use: YOLOv8m (slower, better mAP)
   - Promote best model to MLflow registry

### Skills Learned

- YOLOv8 fine-tuning with Ultralytics
- Understanding detection loss components (box loss, cls loss, dfl loss)
- Multi-scale training for robust detection
- Multi-object tracking with DeepSORT
- Kalman filter basics for motion prediction
- Appearance-based Re-ID for track association

---

## Phase 4: Evaluation

**Duration:** 3-4 days
**Objective:** Thoroughly evaluate detection and tracking quality with proper metrics.

### Tasks

1. **Detection Metrics** -- `src/training/evaluate.py`
   - **mAP@50:** mean Average Precision at IoU threshold 0.50. The standard
     detection metric. A prediction is "correct" if IoU with ground truth > 0.50.
   - **mAP@50:95:** average mAP across IoU thresholds from 0.50 to 0.95 (step 0.05).
     Much harder -- rewards precise localization.
   ```python
   def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
       """Compute Intersection over Union between two boxes.
       Boxes in format [x1, y1, x2, y2]."""
       x1 = max(box1[0], box2[0])
       y1 = max(box1[1], box2[1])
       x2 = min(box1[2], box2[2])
       y2 = min(box1[3], box2[3])
       intersection = max(0, x2 - x1) * max(0, y2 - y1)
       area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
       area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
       union = area1 + area2 - intersection
       return intersection / union if union > 0 else 0.0
   ```
   - **Per-class AP:** which classes does the model detect well vs poorly?
     Small objects (bicycle) are typically harder than large objects (car).
   - **Size-stratified AP:** separately evaluate on small, medium, and large objects.
     This reveals if the model struggles with specific scales.

2. **Speed Benchmarks** -- `src/edge/benchmark.py`
   - Measure FPS across different configurations:
   ```python
   def benchmark_inference(
       model_path: str,
       input_size: int = 640,
       n_warmup: int = 50,
       n_iterations: int = 200,
       device: str = "cuda:0",
   ) -> dict:
       """Benchmark detection inference speed."""
       model = YOLO(model_path)
       dummy_input = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

       # Warmup
       for _ in range(n_warmup):
           model(dummy_input, verbose=False)

       # Benchmark
       times = []
       for _ in range(n_iterations):
           start = time.perf_counter()
           model(dummy_input, verbose=False)
           times.append(time.perf_counter() - start)

       return {
           "mean_latency_ms": np.mean(times) * 1000,
           "p50_latency_ms": np.percentile(times, 50) * 1000,
           "p95_latency_ms": np.percentile(times, 95) * 1000,
           "fps": 1.0 / np.mean(times),
           "device": device,
           "input_size": input_size,
       }
   ```
   - Benchmark comparison table:

   | Runtime | Device | Input Size | FPS | mAP@50 |
   |---------|--------|-----------|-----|--------|
   | PyTorch | GPU | 640 | ~80 | 0.68 |
   | ONNX Runtime | GPU | 640 | ~100 | 0.68 |
   | TensorRT FP16 | GPU | 640 | ~150 | 0.67 |
   | TensorRT INT8 | GPU | 640 | ~200 | 0.65 |
   | ONNX Runtime | CPU | 640 | ~5 | 0.68 |
   | TensorRT FP16 | Jetson Nano | 416 | ~15 | 0.60 |

3. **Tracking Metrics**
   - **MOTA (Multiple Object Tracking Accuracy):** combines false positives,
     false negatives, and identity switches into a single score
   - **MOTP (Multiple Object Tracking Precision):** average IoU between matched
     tracks and ground truth
   - **IDF1:** ratio of correctly identified detections over average of ground
     truth and computed detections. Best metric for identity preservation.
   - **ID Switches:** how often does the tracker lose an identity and reassign it?
   ```python
   def compute_mota(
       gt_tracks: dict[int, list[BBox]],
       pred_tracks: dict[int, list[BBox]],
       iou_threshold: float = 0.5,
   ) -> float:
       """Compute MOTA across all frames."""
       total_fn = 0   # False negatives (missed detections)
       total_fp = 0   # False positives (phantom detections)
       total_idsw = 0  # Identity switches
       total_gt = 0   # Total ground truth objects

       for frame_id in gt_tracks:
           gt_boxes = gt_tracks[frame_id]
           pred_boxes = pred_tracks.get(frame_id, [])
           total_gt += len(gt_boxes)
           # Match predictions to ground truth using Hungarian algorithm
           matches, fn, fp, idsw = match_frame(gt_boxes, pred_boxes, iou_threshold)
           total_fn += fn
           total_fp += fp
           total_idsw += idsw

       mota = 1 - (total_fn + total_fp + total_idsw) / total_gt
       return mota
   ```

4. **Error Analysis** -- `notebooks/04_evaluation.ipynb`
   - Visualize false positives: what is the model detecting that is not there?
   - Visualize false negatives: what is the model missing?
   - Analyze by: object size, occlusion level, class, image brightness
   - Common failure modes: small objects, heavily occluded objects, objects at
     image edges, unusual viewpoints

5. **Generate Evaluation Report**
   - Comprehensive comparison: per-class AP, size-stratified AP, speed benchmarks
   - Precision-recall curves per class
   - Confusion matrix: which classes get confused with each other?
   - Log all artifacts to MLflow

### Skills Learned

- Detection evaluation metrics (mAP, IoU, per-class AP)
- Tracking evaluation metrics (MOTA, MOTP, IDF1)
- Speed benchmarking methodology for real-time systems
- Error analysis for object detection (size, occlusion, class confusion)
- Understanding the accuracy vs speed tradeoff

---

## Phase 5: API & Serving

**Duration:** 4-5 days
**Objective:** Build a detection API with both REST (images) and WebSocket (video streams) support.

### Tasks

1. **Define API Schemas** -- `src/serving/schemas.py`
   ```python
   class BoundingBox(BaseModel):
       x1: float
       y1: float
       x2: float
       y2: float
       confidence: float
       class_name: str
       class_id: int
       track_id: int | None = None  # Only for tracked objects

   class DetectionResponse(BaseModel):
       detections: list[BoundingBox]
       n_objects: int
       inference_time_ms: float
       model_version: str
       image_size: tuple[int, int]

   class VideoFrameResponse(BaseModel):
       frame_number: int
       timestamp_ms: float
       tracked_objects: list[BoundingBox]
       fps: float

   class BatchDetectionRequest(BaseModel):
       confidence_threshold: float = 0.25
       iou_threshold: float = 0.45
       target_classes: list[str] | None = None
   ```

2. **Single Image Detection Endpoint** -- `src/serving/detect.py`
   - `POST /detect` -- upload an image, get back detections
   - Apply NMS (Non-Maximum Suppression) to remove duplicate boxes
   - Optionally return the annotated image with boxes drawn
   ```python
   @app.post("/detect", response_model=DetectionResponse)
   async def detect_objects(
       file: UploadFile,
       confidence: float = Query(0.25, ge=0.0, le=1.0),
       iou_threshold: float = Query(0.45, ge=0.0, le=1.0),
       draw_boxes: bool = Query(False),
   ):
       image_bytes = await file.read()
       image = decode_image(image_bytes)
       start = time.perf_counter()
       results = model(image, conf=confidence, iou=iou_threshold, verbose=False)[0]
       latency = (time.perf_counter() - start) * 1000

       detections = []
       for box in results.boxes:
           detections.append(BoundingBox(
               x1=float(box.xyxy[0][0]),
               y1=float(box.xyxy[0][1]),
               x2=float(box.xyxy[0][2]),
               y2=float(box.xyxy[0][3]),
               confidence=float(box.conf[0]),
               class_name=model.names[int(box.cls[0])],
               class_id=int(box.cls[0]),
           ))

       if draw_boxes:
           annotated = results.plot()
           return StreamingResponse(encode_image(annotated), media_type="image/jpeg")

       return DetectionResponse(
           detections=detections,
           n_objects=len(detections),
           inference_time_ms=latency,
           model_version=MODEL_VERSION,
           image_size=(image.shape[1], image.shape[0]),
       )
   ```

3. **Video Stream with WebSocket** -- `src/serving/stream.py`
   - WebSocket endpoint for real-time video processing
   - Client sends video frames, server returns detections + tracking
   ```python
   @app.websocket("/ws/stream")
   async def video_stream(websocket: WebSocket):
       await websocket.accept()
       tracker = ObjectTracker(detector=model)
       frame_count = 0
       fps_counter = FPSCounter()

       try:
           while True:
               # Receive frame from client
               data = await websocket.receive_bytes()
               frame = decode_frame(data)
               frame_count += 1

               # Detect and track
               tracked_objects = tracker.process_frame(frame)
               fps = fps_counter.update()

               # Send back results
               response = VideoFrameResponse(
                   frame_number=frame_count,
                   timestamp_ms=time.time() * 1000,
                   tracked_objects=[obj.to_schema() for obj in tracked_objects],
                   fps=fps,
               )
               await websocket.send_json(response.model_dump())
       except WebSocketDisconnect:
           logger.info(f"Client disconnected after {frame_count} frames")
   ```

4. **Batch Processing Endpoint** -- `src/serving/batch.py`
   - `POST /detect/batch` -- upload a ZIP of images, get back all detections
   - Process images in batches for GPU efficiency
   - Return results as JSON or annotated images in a ZIP

5. **Visualization Utilities** -- `src/serving/visualize.py`
   - Draw bounding boxes with class labels and confidence scores
   - Draw track IDs and motion trails for tracked objects
   - Color-code by class or track ID
   ```python
   def draw_detections(
       frame: np.ndarray,
       detections: list[BoundingBox],
       draw_trails: bool = False,
       trail_history: dict[int, list[tuple]] | None = None,
   ) -> np.ndarray:
       """Draw bounding boxes and labels on frame."""
       annotated = frame.copy()
       for det in detections:
           color = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]
           cv2.rectangle(annotated,
               (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)),
               color, 2)
           label = f"{det.class_name} {det.confidence:.2f}"
           if det.track_id is not None:
               label = f"ID:{det.track_id} {label}"
           cv2.putText(annotated, label,
               (int(det.x1), int(det.y1) - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
       return annotated
   ```

6. **Build Streamlit Frontend** -- `src/frontend/app.py`
   - Image upload with detection visualization
   - Video upload with tracking visualization
   - Adjustable confidence and IoU thresholds via sliders
   - Real-time FPS display
   - Class filter checkboxes

7. **API Tests**
   - Unit tests: bounding box schema, NMS, frame encoding/decoding
   - Integration tests: full detection pipeline, WebSocket stream, batch processing
   - Performance tests: verify FPS targets with concurrent WebSocket connections

### Skills Learned

- WebSocket streaming for real-time ML (not just REST)
- Video frame processing and encoding
- NMS (Non-Maximum Suppression) for detection post-processing
- Batch inference for GPU efficiency
- Building real-time visualization overlays

---

## Phase 6: Edge Deployment

**Duration:** 4-5 days
**Objective:** Optimize the model for edge devices and deploy on Jetson Nano or Raspberry Pi.

### Tasks

1. **ONNX Export** -- `src/model/export.py`
   - Export YOLOv8 to ONNX format for portable inference
   ```python
   def export_to_onnx(model_path: str, imgsz: int = 640) -> str:
       """Export YOLOv8 to ONNX format."""
       model = YOLO(model_path)
       onnx_path = model.export(
           format="onnx",
           imgsz=imgsz,
           simplify=True,     # Simplify ONNX graph
           opset=12,          # ONNX opset version
           dynamic=False,     # Static input shape for TensorRT
       )
       return onnx_path
   ```

2. **TensorRT Optimization** -- `src/edge/tensorrt_engine.py`
   - Convert ONNX to TensorRT engine for maximum GPU inference speed
   - Support FP16 (half precision) and INT8 (quantized) modes
   ```python
   def build_tensorrt_engine(
       onnx_path: str,
       engine_path: str,
       precision: str = "fp16",  # "fp32", "fp16", or "int8"
       max_batch_size: int = 1,
       workspace_size: int = 1 << 30,  # 1 GB
   ) -> None:
       """Build TensorRT engine from ONNX model."""
       model = YOLO(onnx_path)
       model.export(
           format="engine",
           half=(precision == "fp16"),
           int8=(precision == "int8"),
           workspace=workspace_size / (1 << 30),
           batch=max_batch_size,
       )
       logger.info(f"TensorRT engine saved to {engine_path}")
   ```

3. **INT8 Quantization** -- `src/edge/quantize.py`
   - INT8 quantization needs a calibration dataset (representative subset of training data)
   - Measure accuracy loss: typically < 2% mAP drop for 2-3x speedup
   - Compare: FP32 vs FP16 vs INT8 on accuracy and speed

4. **Edge Inference Engine** -- `src/edge/tensorrt_engine.py`
   - Lightweight inference wrapper for edge devices
   - Handle: model loading, preprocessing, NMS, all in optimized pipeline
   ```python
   class EdgeDetector:
       def __init__(self, engine_path: str, conf_threshold: float = 0.25):
           self.model = YOLO(engine_path, task="detect")
           self.conf_threshold = conf_threshold

       def detect(self, frame: np.ndarray) -> list[Detection]:
           results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
           return self._parse_results(results)

       def detect_stream(self, source: str) -> Generator[list[Detection], None, None]:
           """Process video stream (camera, RTSP, file)."""
           results = self.model(source, stream=True, conf=self.conf_threshold)
           for result in results:
               yield self._parse_results(result)
   ```

5. **Jetson Nano Deployment** -- `edge/jetson_setup.sh`
   - Install JetPack SDK, TensorRT, and Python dependencies
   - Deploy TensorRT engine to Jetson Nano
   - Connect USB camera or RTSP stream
   - Target: 15+ FPS at 416x416 input resolution
   ```bash
   #!/bin/bash
   # Jetson Nano setup for object detection
   # Requires JetPack 4.6+ installed

   # Install Python packages
   pip3 install ultralytics opencv-python-headless numpy

   # Copy TensorRT engine to device
   scp model.engine jetson@192.168.1.100:/home/jetson/models/

   # Run detection on camera
   python3 run_edge.py --source 0 --model /home/jetson/models/model.engine
   ```

6. **Raspberry Pi Deployment** (alternative to Jetson)
   - Use ONNX Runtime (no TensorRT on RPi)
   - Much slower (2-5 FPS) but demonstrates CPU-only edge deployment
   - Useful for understanding the gap between GPU and CPU edge inference

7. **Speed/Accuracy Benchmark** -- `src/edge/benchmark.py`
   - Comprehensive benchmark across all deployment targets
   - Generate benchmark report with speed/accuracy tradeoff curves
   - Plot: mAP vs FPS for each configuration

### Skills Learned

- ONNX model export and optimization
- TensorRT engine building (FP16, INT8 quantization)
- Edge deployment on NVIDIA Jetson
- Understanding inference optimization tradeoffs (accuracy vs speed vs memory)
- Camera and RTSP stream processing
- Cross-platform model deployment

---

## Phase 7: Containerization

**Duration:** 2-3 days
**Objective:** Package the detection system into Docker containers.

### Tasks

1. **API Dockerfile** -- `docker/Dockerfile.api`
   - Use NVIDIA CUDA base image for GPU support
   - Multi-stage build to keep the image size manageable
   ```dockerfile
   # Builder stage
   FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS builder
   RUN apt-get update && apt-get install -y python3-pip libgl1-mesa-glx libglib2.0-0
   WORKDIR /app
   COPY pyproject.toml .
   RUN pip3 install --no-cache-dir .

   # Runtime stage
   FROM nvidia/cuda:12.1-runtime-ubuntu22.04
   RUN apt-get update && apt-get install -y python3 libgl1-mesa-glx libglib2.0-0
   COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
   COPY src/ /app/src/
   COPY models/ /app/models/
   EXPOSE 8000
   CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Edge Dockerfile** -- `docker/Dockerfile.edge`
   - Minimal container for edge deployment
   - Based on NVIDIA L4T (Linux for Tegra) for Jetson devices
   - Include only inference dependencies (no training libraries)

3. **docker-compose.yaml**
   ```yaml
   services:
     api:
       build:
         context: .
         dockerfile: docker/Dockerfile.api
       ports: ["8000:8000"]
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
       environment:
         - MODEL_PATH=/app/models/yolov8s.pt
         - CONFIDENCE_THRESHOLD=0.25

     frontend:
       build:
         context: .
         dockerfile: docker/Dockerfile.frontend
       ports: ["8501:8501"]
       depends_on: [api]

     prometheus:
       image: prom/prometheus
       volumes: ["./prometheus:/etc/prometheus"]
       ports: ["9090:9090"]

     grafana:
       image: grafana/grafana
       volumes: ["./grafana:/etc/grafana/provisioning"]
       ports: ["3000:3000"]
       depends_on: [prometheus]
   ```

4. **Verify the full stack**
   - `docker compose up` -- all services start
   - Test image detection via REST API
   - Test video streaming via WebSocket
   - Verify GPU passthrough is working inside the container
   - Prometheus scrapes metrics, Grafana shows dashboard

### Skills Learned

- GPU-enabled Docker containers (NVIDIA runtime)
- Multi-stage builds for CV applications (OpenCV dependencies)
- Container size optimization for edge deployment
- GPU passthrough in Docker Compose

---

## Phase 8: Testing & CI/CD

**Duration:** 2-3 days
**Objective:** Automate quality checks for the detection pipeline.

### Tasks

1. **Write comprehensive tests**

   **Unit tests:**
   ```
   test_convert.py
   ├── test_coco_to_yolo_format_conversion
   ├── test_coordinate_normalization
   └── test_class_id_mapping

   test_augment.py
   ├── test_mosaic_output_shape
   ├── test_boxes_within_image_after_augmentation
   ├── test_horizontal_flip_mirrors_boxes
   └── test_augmented_box_areas_positive

   test_detector.py
   ├── test_detection_output_schema
   ├── test_nms_removes_overlapping_boxes
   ├── test_confidence_filtering
   └── test_model_handles_different_input_sizes

   test_tracker.py
   ├── test_track_id_persistence_across_frames
   ├── test_new_object_gets_new_id
   └── test_lost_track_removal_after_max_age
   ```

   **Integration tests:**
   ```
   test_api.py
   ├── test_detect_returns_bounding_boxes
   ├── test_detect_with_annotated_image
   ├── test_batch_detect_multiple_images
   ├── test_invalid_image_returns_422
   └── test_confidence_threshold_filtering

   test_stream.py
   ├── test_websocket_connection
   ├── test_websocket_receives_tracking_data
   └── test_websocket_handles_disconnection
   ```

   **ML-specific tests:**
   ```
   ├── test_mAP_above_threshold_on_val_set
   ├── test_inference_fps_above_30
   ├── test_onnx_output_matches_pytorch
   └── test_tensorrt_output_matches_onnx
   ```

2. **CI Pipeline** -- `.github/workflows/ci.yaml`
   - Lint, type check, run unit tests (CPU-only, no GPU in CI)
   - Build Docker images
   - Smoke test: start API, send test image, verify detection response schema
   - Note: full mAP evaluation runs nightly (too slow for every PR)

3. **CD Pipeline** -- `.github/workflows/cd.yaml`
   - Build and push Docker images on merge to main
   - Deploy to staging, run integration tests
   - Deploy to production on manual approval

### Skills Learned

- Testing real-time ML systems
- Testing WebSocket endpoints
- ML model regression testing (mAP thresholds)
- CI/CD for GPU-dependent applications

---

## Phase 9: Monitoring

**Duration:** 2-3 days
**Objective:** Monitor detection quality, system performance, and object statistics in production.

### Tasks

1. **System Metrics** -- `src/monitoring/metrics.py`
   - `detection_requests_total` -- counter by endpoint type (image/stream/batch)
   - `detection_latency_seconds` -- histogram (critical for real-time)
   - `active_websocket_connections` -- gauge
   - `gpu_utilization_percent` -- gauge
   - `gpu_memory_used_bytes` -- gauge

2. **FPS Tracking** -- `src/monitoring/fps_tracker.py`
   - Real-time FPS monitoring for video streams
   - Track: current FPS, average FPS, min FPS (frame drops)
   - Alert if FPS drops below threshold (system cannot keep up)
   ```python
   class FPSTracker:
       def __init__(self, window_size: int = 30):
           self.timestamps: deque[float] = deque(maxlen=window_size)

       def update(self) -> float:
           now = time.perf_counter()
           self.timestamps.append(now)
           if len(self.timestamps) < 2:
               return 0.0
           elapsed = self.timestamps[-1] - self.timestamps[0]
           return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0
   ```

3. **Detection Quality Monitoring** -- `src/monitoring/class_distribution.py`
   - Track detected class distribution over time
   - If the model suddenly detects 90% "person" when it normally detects a mix,
     something may have changed (camera angle, scene, model issue)
   - Track confidence score distribution per class
   - Monitor for confidence degradation over time

4. **Grafana Dashboard** -- `grafana/dashboards/detection_monitoring.json`
   - Row 1: Request rate, latency percentiles, active streams, error rate
   - Row 2: FPS per stream, GPU utilization, GPU memory
   - Row 3: Detected objects per class over time, confidence distribution
   - Row 4: Object count per frame distribution, detection size distribution

5. **Alerting Rules**
   - FPS < 15 for any stream for 30 seconds -> alert
   - GPU utilization > 95% for 5 minutes -> alert (may need to limit concurrent streams)
   - Error rate > 5% -> alert
   - Confidence distribution shift > 2 std deviations -> alert
   - GPU memory > 90% -> alert

### Skills Learned

- Real-time performance monitoring (FPS, GPU utilization)
- Detection-specific quality monitoring
- Class distribution drift detection
- Alerting for real-time systems

---

## Timeline Summary

```
Week 1   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 1: Setup & Design Doc    (2 days)
         Phase 2: Data Pipeline         (3 days)

Week 2   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 2: Data Pipeline cont.   (2 days)
         Phase 3: Model Development     (3 days)

Week 3   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 3: Model Development     (4 days)
         Phase 4: Evaluation            (1 day)

Week 4   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 4: Evaluation cont.      (3 days)
         Phase 5: API & Serving         (2 days)

Week 5   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 5: API & Serving cont.   (3 days)
         Phase 6: Edge Deployment       (2 days)

Week 6   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 6: Edge Deployment       (3 days)
         Phase 7: Containerization      (2 days)

Week 7   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 7: Containerization      (1 day)
         Phase 8: Testing & CI/CD       (3 days)
         Phase 9: Monitoring            (1 day)

Week 8   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 9: Monitoring cont.      (2 days)
         Buffer / catch-up              (3 days)
```

**Total: ~40 days (8 weeks at a comfortable pace)**

---

## Skills Checklist

When you complete this project, you will have hands-on experience with:

- [ ] Understanding object detection vs classification
- [ ] Working with COCO dataset and annotation formats
- [ ] Annotation format conversion (COCO to YOLO)
- [ ] Detection-specific EDA (box sizes, aspect ratios, class distribution)
- [ ] Detection-aware augmentation (mosaic, mixup with bounding box transforms)
- [ ] YOLOv8 fine-tuning with Ultralytics
- [ ] Multi-scale training for robust detection
- [ ] Detection loss components (box, classification, distribution focal loss)
- [ ] Multi-object tracking with DeepSORT
- [ ] Detection metrics (mAP@50, mAP@50:95, per-class AP)
- [ ] Tracking metrics (MOTA, MOTP, IDF1)
- [ ] IoU computation and Non-Maximum Suppression
- [ ] Speed benchmarking for real-time systems
- [ ] ONNX model export and optimization
- [ ] TensorRT engine building (FP16, INT8 quantization)
- [ ] Edge deployment (Jetson Nano or Raspberry Pi)
- [ ] WebSocket streaming for real-time video processing
- [ ] FastAPI with both REST and WebSocket endpoints
- [ ] Video frame processing with OpenCV
- [ ] GPU-enabled Docker containers
- [ ] Real-time FPS and GPU monitoring
- [ ] Detection quality monitoring (class distribution drift)
- [ ] CI/CD for GPU-dependent applications
- [ ] Experiment tracking with MLflow

---

## Key Differences from Classification Projects

| Concept | Classification | Object Detection |
|---------|---------------|-----------------|
| Output | Single label per image | Multiple boxes + labels per image |
| Loss function | Cross-entropy | Box regression + classification + objectness |
| Evaluation | Accuracy, F1 | mAP@50, mAP@50:95 (IoU-based) |
| Data format | Image -> label | Image -> list of (class, x, y, w, h) |
| Augmentation | Transform image only | Transform image AND bounding boxes |
| Real-time | One prediction per request | Continuous video stream processing |
| Serving | REST API only | REST + WebSocket for streaming |
| Optimization | ONNX is nice-to-have | TensorRT is essential for real-time |
| Edge deployment | Not typically needed | Critical for many use cases |
| Infrastructure | CPU often sufficient | GPU required for real-time |

---

## Getting Started

Ready to begin? Start with Phase 1:

```bash
# 1. Initialize the project
mkdir object-detection-tracking && cd object-detection-tracking
git init

# 2. Create the folder structure
mkdir -p configs data/{raw,processed,augmented,custom} notebooks \
  src/{data,model,training,serving,edge,monitoring,frontend} \
  tests/{unit,integration} docker .github/workflows \
  grafana/dashboards prometheus edge scripts

# 3. Verify GPU and install Ultralytics
pip install ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 4. Start writing DESIGN_DOC.md
```
