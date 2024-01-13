
# AI-Driven Stock Price Prediction Using TensorFlow

## Project Overview
This project focuses on predicting stock prices using deep learning techniques in TensorFlow. The core of this project involves parsing historical stock price data and employing a Long Short-Term Memory (LSTM) neural network to learn and predict future prices. The primary dataset used is the Bitcoin USD prices from Yahoo Finance.

## Objective
The objective of this project is to explore and demonstrate the capabilities of LSTM neural networks in forecasting stock prices, providing insights into the potential of AI in financial analysis and prediction.

## Prerequisites
It's essential to install TensorFlow 2.11.0 +, particularly the GPU version for better performance:
```bash
!pip install tensorflow-gpu==2.11.0
```

## Key Technologies
- TensorFlow and Keras for building and training the LSTM model.
- Pandas and Numpy for data manipulation.
- Matplotlib and Seaborn for data visualization.
- Scikit-learn for additional machine learning utilities.

## How It Works
1. **Data Preparation**: The Bitcoin USD historical data is loaded and preprocessed.
2. **Model Building**: A Sequential LSTM model is built using TensorFlow and Keras.
3. **Training**: The model is trained on the historical data.
4. **Evaluation**: The model's performance is evaluated using the test dataset.
5. **Prediction**: Future stock prices are predicted based on the learned patterns.

## Example Visualization
After training and evaluating the model, visualizations of the predicted versus actual stock prices can be generated, showcasing the model's forecasting ability.
