# Google Stock Price Prediction

## Overview

This project demonstrates the use of Long Short-Term Memory (LSTM) networks for predicting Google stock prices. The model is trained on historical stock prices and makes predictions on test data. The project includes data preprocessing, LSTM model training, and result visualization.

## Project Components

### 1. Data Preparation

**Files**:
- `Google_Stock_Price_Train.csv`: Training dataset containing historical stock prices.
- `Google_Stock_Price_Test.csv`: Test dataset used to evaluate the model.

**Steps**:
- Load the training and test datasets.
- Normalize the training data using Min-Max scaling.
- Create time series data with 60 timesteps for training.
- Prepare the test data for prediction.

### 2. Model Architecture (`model.py`)

**Purpose**: Define and train an LSTM model for stock price prediction.

**Techniques and Tools**:
- **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) well-suited for time series prediction.
- **Dropout**: Regularization technique used to prevent overfitting.
- **Keras**: High-level API for building and training neural networks.

**Model Structure**:
- **LSTM Layers**: Four LSTM layers with 50 units each, with Dropout layers to prevent overfitting.
- **Dense Layer**: Output layer with a single neuron for prediction.
- **Optimizer**: Adam optimizer.
- **Loss Function**: Mean Squared Error (MSE).

### 3. Training the Model

**Steps**:
1. Prepare the training data with 60 timesteps and the corresponding output.
2. Train the LSTM model for 100 epochs with a batch size of 32.

### 4. Making Predictions

**Steps**:
1. Load the test dataset.
2. Prepare the test data similar to the training data.
3. Use the trained model to predict stock prices.
4. Inverse-transform the predicted values to get actual stock prices.

### 5. Results Visualization

**Plot**:
- Compare the real Google stock prices with the predicted values.
- Display the plot with `matplotlib`.

## How It Works

### 1. Data Preparation

1. **Load Data**:
   - Load `Google_Stock_Price_Train.csv` and `Google_Stock_Price_Test.csv`.

2. **Normalize Data**:
   - Apply Min-Max scaling to the training set.

3. **Create Time Series Data**:
   - Generate sequences with 60 timesteps for the LSTM model.

### 2. Model Training

1. **Define Model**:
   - Initialize an LSTM model with four LSTM layers and Dropout layers.
   - Compile the model with the Adam optimizer and MSE loss function.

2. **Train Model**:
   - Fit the model on the training data for 100 epochs.

### 3. Predictions

1. **Prepare Test Data**:
   - Normalize and reshape test data similar to the training data format.

2. **Predict**:
   - Use the trained model to predict stock prices on the test set.

3. **Inverse Transform**:
   - Convert the predicted values back to the original scale.

### 4. Visualization

1. **Plot Results**:
   - Plot real vs. predicted stock prices using `matplotlib`.

