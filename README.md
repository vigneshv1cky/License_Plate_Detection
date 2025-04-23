


# License Plate Detection using YOLOv8 🚗🔍

This project leverages **YOLOv8**, a cutting-edge object detection model, to detect license plates in images and videos. The solution supports both training and inference using a custom dataset sourced from Roboflow. It is implemented in a Jupyter Notebook and includes detailed explanations, visual outputs, and performance metrics.

## 🧠 Project Structure

### 1. **Setup and Installation**
- Installs required libraries such as `ultralytics`, `opencv-python`, and `matplotlib`.
- Verifies YOLOv8 installation and ensures the environment is ready for training.

### 2. **Data Preparation**
- Dataset is downloaded from [Roboflow](https://universe.roboflow.com/) and unzipped into a local directory.
- `data.yaml` is used to define class names and dataset paths.
- Images and labels are structured into `train`, `valid`, and `test` folders.

### 3. **Model Training**
- Trains YOLOv8 on the custom license plate dataset.
- Configurable parameters include:
  - `model = YOLO("yolov8n.pt")` (can be changed to `s`, `m`, `l`, or `x` variants)
  - `epochs = 10`
  - `imgsz = 640`

### 4. **Validation and Testing**
- Loads the best weights after training: `runs/detect/train/weights/best.pt`.
- Evaluates model performance on validation and test sets.
- Visualizes bounding boxes on images with predictions.

### 5. **Custom Inference**
- Performs inference on:
  - Images
  - Videos
  - Real-time webcam stream (optional)
- Saves output images with bounding boxes to an output directory.

---

## 📊 Sample Results

Sample output shows bounding boxes over detected license plates with confidence scores:

```python
result = model.predict(source="test/image.jpg", save=True)
```

Output images are saved in `runs/detect/predict`.

---

## 📁 Directory Structure

```
├── data.yaml
├── train/
├── valid/
├── test/
├── License_plate_detection.ipynb
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt
```

---

## 💡 Future Enhancements

- Fine-tuning with more diverse datasets for generalization.
- Integrating with OCR models to extract license numbers.
- Deploying as a Streamlit or Flask web app.

---

## 📚 References

- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [Roboflow Datasets](https://universe.roboflow.com/)
- [Ultralytics Documentation](https://docs.ultralytics.com)

---

## 🙌 Acknowledgements

Thanks to [Roboflow](https://roboflow.com/) for dataset hosting and [Ultralytics](https://ultralytics.com/) for the YOLOv8 framework.

---

## 🛠️ Requirements

```bash
pip install ultralytics opencv-python matplotlib
```

---

## 🔗 Run Notebook

To run the notebook:
1. Clone the repo or upload the `.ipynb` file to Google Colab.
2. Ensure dataset is uploaded or downloaded via the Roboflow link.
3. Run all cells in order for training and inference.

---

## 🔒 License

This project is licensed under the [MIT License](LICENSE).


