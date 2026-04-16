# Face Mask Detection System

An end-to-end real-time face mask detection system built with **YOLOv8**, **FastAPI**, and **Streamlit**, fully containerized with **Docker** for consistent deployment.

The model detects three classes:
- `with_mask`
- `without_mask`
- `mask_weared_incorrect`

---

## Results

The model was trained on the [Face Mask Detection dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) (853 images) using **YOLOv8s** for 72 epochs with early stopping on an NVIDIA RTX 3060.

| Metric          | Score  |
|-----------------|--------|
| **Precision**   | 95.6%  |
| **Recall**      | 85.6%  |
| **mAP@0.5**     | 89.4%  |
| **mAP@0.5:0.95**| 62.3%  |

Training graphs, confusion matrix, and PR curves are available in `runs/detect/face_mask_v2/`.

---

## Tech Stack

- **Model**: YOLOv8s (Ultralytics)
- **Backend**: FastAPI
- **Frontend / UI**: Streamlit
- **Containerization**: Docker
- **Language**: Python 3.9
- **Dataset**: Kaggle Face Mask Detection (853 images, 3 classes)

---

## Project Structure

```
FaceMask_FullProject/
├── app/
│   ├── main.py              # FastAPI inference server
│   └── best.pt              # Trained YOLOv8s model
├── dataset/                 # Prepared YOLO-format dataset (gitignored)
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── data.yaml
├── runs/                    # Training output (gitignored)
├── prepare_dataset.py       # XML to YOLO converter + train/val split
├── train.py                 # Training script
├── ui.py                    # Streamlit UI
├── data.yaml                # Dataset config
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container build
└── README.md
```

---

## Installation

### Option 1: Local Setup

```bash
git clone https://github.com/Beshoy-Atef-Adel/Face-Mask-Detection-System.git
cd Face-Mask-Detection-System

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Option 2: Docker

```bash
docker build -t face-mask-detection .
docker run -p 8000:8000 face-mask-detection
```

---

## Usage

### Run the FastAPI inference server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

**Endpoint**: `POST /predict`
- **Body**: multipart/form-data with `file` field containing an image
- **Response**: JSON with detection results

### Run the Streamlit UI

In a separate terminal:

```bash
streamlit run ui.py
```

Then open `http://localhost:8501` in your browser, upload an image, and click **Predict**.

---

## Training (Reproducing the Results)

### Step 1: Download the dataset
Download the [Kaggle Face Mask Detection dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) and place it in `data/` so you have:
```
data/
├── images/        # 853 PNG files
└── annotations/   # 853 XML files (Pascal VOC format)
```

### Step 2: Convert XML to YOLO format and split
```bash
python prepare_dataset.py
```
This creates `dataset/` with proper YOLO structure (682 train / 171 val).

### Step 3: Train
```bash
python train.py
```

Default config: YOLOv8s, 100 epochs, batch=8, imgsz=640, patience=30 (early stopping).

Training takes about 25–35 minutes on an RTX 3060.

---

## Dataset

| Class                    | Object Count |
|--------------------------|--------------|
| `with_mask`              | 3,232        |
| `without_mask`           | 717          |
| `mask_weared_incorrect`  | 123          |

**Train/Val split**: 80/20 (682 / 171 images)

---

## Author

**Beshoy Atef Adel**
AI Engineer | Computer Vision & MLOps
- GitHub: [@Beshoy-Atef-Adel](https://github.com/Beshoy-Atef-Adel)
- LinkedIn: [beshoy-atef-adel2002](https://linkedin.com/in/beshoy-atef-adel2002)
- Kaggle: [beshoyatefadel](https://www.kaggle.com/beshoyatefadel)
