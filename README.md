# Malaria Cell Image Classification with CNN

Deep learning project that detects malaria parasites in thin blood smear cell images.

- **Task:** binary image classification (Parasitized vs. Uninfected)
- **Dataset:** 27,558 cell images, perfectly balanced (13,779 per class)
- **Approach:** custom CNN built from scratch vs. transfer learning with VGG16

## Workflow

1. Load and explore images, check class balance
2. Preprocess: resize to 128x128 (custom CNN) / 224x224 (transfer learning), normalize pixels
3. Train a custom CNN (2 Conv + MaxPooling blocks, Dense head, sigmoid output)
4. Train a transfer learning model: frozen VGG16 (ImageNet weights) + custom classifier head
5. Compare validation performance

## Results

| Model | Validation Accuracy |
| :--- | :---: |
| Custom CNN | ~94% (best 94.6%) |
| VGG16 Transfer Learning | ~93% (best 93.2%) |

The custom CNN reaches near-perfect training accuracy but validation accuracy plateaus around 94%, which shows overfitting. Possible next steps: data augmentation, dropout, early stopping, fine-tuning deeper layers.

## Tech Stack

Python, TensorFlow/Keras, OpenCV, NumPy, pandas, scikit-learn, Matplotlib, Seaborn

## Files

- `CNN_Malaria_Data.ipynb`: full training and evaluation notebook

## Data

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria). It is not included in this repository due to its size. Download it and place the `Parasitized` and `Uninfected` folders inside a `Malaria/` directory in the project root.
