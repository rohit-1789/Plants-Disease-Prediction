# Plant Disease Prediction System

A deep learning-based system to detect plant diseases using images of crop leaves. This project leverages convolutional neural networks (CNNs) to classify plant diseases into 38 categories, ensuring accurate and efficient disease recognition to aid farmers and researchers.

---

## Features

- **Accurate Detection:** Utilizes advanced CNN architectures to identify diseases with high precision.
- **User-Friendly Interface:** Provides an intuitive Streamlit web application for disease recognition.
- **Real-Time Predictions:** Upload images to receive disease detection results instantly.
- **Visualization:** Displays training progress, accuracy trends, and confusion matrix for better understanding.

---

## Dataset

The dataset contains images of healthy and diseased crop leaves across 38 classes. The data is divided as follows:

- **Training Set:** 37,940 images
- **Validation Set:** 17,572 images
- **Test Set:** 33 images

Each image is resized to `128x128` for uniformity during training and inference.

---

## Installation

### Prerequisites

1. Python 3.7+
2. TensorFlow
3. Streamlit
4. Required libraries (install via `requirements.txt`)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-disease-prediction.git
   cd plant-disease-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Organize training and validation data in `train/` and `valid/` directories.

4. Run the application:
   ```bash
   streamlit run main.py
   ```

---

## How to Use

### Using the Streamlit App

1. **Home Page:** Provides an overview of the system and its features.
2. **About Page:** Offers information about the dataset and project details.
3. **Disease Recognition Page:**
   - Upload an image of a plant leaf.
   - Click `Predict` to classify the disease.
   - View the prediction result and corresponding class name.

---

## Model Training

The CNN model is trained using TensorFlow, with the following architecture:

1. **Convolutional Layers:** Extract features from images.
2. **Pooling Layers:** Reduce spatial dimensions.
3. **Dropout Layers:** Prevent overfitting.
4. **Fully Connected Layers:** Map features to output classes.

### Training Configuration:
- Optimizer: Adam
- Learning Rate: 0.0001
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy

Run the training script:
```bash
python train_model.py
```

---

## Results and Visualization

1. **Accuracy Trends:**
   - Training vs. Validation accuracy plotted over epochs.
2. **Confusion Matrix:**
   - Visualized using Seaborn heatmap for detailed insights.

---

## File Structure

```
plant-disease-prediction/
|├── main.py               # Streamlit app
|├── train_model.py       # Model training script
|├── trained_plant_disease_model.keras  # Trained model
|├── training_hist.json   # Training history
|├── requirements.txt     # Dependencies
|└── README.md            # Project documentation
```

---

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any feature additions or bug fixes.


## Acknowledgments

- Dataset: Original dataset sourced from [Kaggle](https://www.kaggle.com).
- Tools: TensorFlow, Streamlit, and other Python libraries.
- Inspiration: Promoting sustainable agriculture through technology.

---

## Contact

For questions or feedback, please contact [abishekahss12@gmail.com].

