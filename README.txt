# CELM: Cardiomegaly Classification from Chest X-rays

This repository provides all components needed to train, evaluate, and deploy deep learning models for binary classification of cardiomegaly (enlarged heart) using chest radiographs (CXRs). The core of the project is an ensemble model (CELM) combining VGG16, ResNet50, and InceptionV3 features with a meta-classifier for robust and accurate diagnosis.

## 💾 Dataset Summary

The dataset is built from curated and balanced subsets of four major public CXR datasets:

- **CheXpert**: 23,002 cardiomegaly, 7,869 normal
- **NIH ChestX-ray14**: 1,563 cardiomegaly, 1,563 normal
- **PadChest**: 2,140 cardiomegaly, 8,708 normal
- **VinDr-CXR**: 230 cardiomegaly, 1,003 normal

> 📦 Combined total: **46,078 samples**  
> 📄 `combined_dataset.csv` includes columns: `image_id`, `label`, `dataset`

## 🧠 Models Included

- VGG16
- ResNet50
- InceptionV3
- DenseNet121, DenseNet201
- AlexNet (custom)
- ViT-B/16 (Vision Transformer)
- Custom CNN
- CELM (Ensemble of VGG16 + ResNet50 + InceptionV3)

## 🔧 Training

Each model is trained using TensorFlow/Keras with:

- Binary Crossentropy or Softmax for classification
- Data augmentation (CLAHE, flips, brightness, rotation)
- Training scripts in `train_all_models.py`, `train_celm.py`
- Exports to `.h5`, `SavedModel/`, and `.tflite`

## 🧪 Evaluation

Evaluation includes:

- Accuracy, F1-score, Precision, Recall
- Confusion matrix, classification report
- Training plots per model
- Inference script: `inference.py`

## 📦 Deployment

All models can be exported and deployed using:

```bash
python export_all_models.py
```

Or use individual scripts like `export_vgg16.py`, `export_resnet50.py`, etc.

## 📜 License

This project is licensed under the [MIT License](./LICENSE).

