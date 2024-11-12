Iris Flower Classification with K-Nearest Neighbors (KNN)
Project Overview
This project involves building a K-Nearest Neighbors (KNN) classification model to classify Iris flowers into three species:
Setosa
Versicolor
Virginica

The model uses the Iris dataset, a well-known dataset in machine learning from Kaggle, containing four features:
Sepal Length
Sepal Width
Petal Length
Petal Width

The goal is to classify the species of an Iris flower based on these features.

Dataset
The Iris dataset contains:
150 samples
3 classes (Setosa, Versicolor, Virginica)
4 features per sample (Sepal length, Sepal width, Petal length, Petal width)

Installation
To run this project, ensure that you have Python installed and install the following libraries:

pip install scikit-learn, seaborn, matplotlib and joblib

Model Training
Load the dataset: Import the Iris dataset from scikit-learn.
Preprocess the data: Split the dataset into training and testing sets (e.g., 80% training and 20% testing).
Train the KNN model: Use the K-Nearest Neighbors algorithm to train the model on the training data.
Evaluate the model: Test the model on the test dataset and calculate accuracy.



Loading the Model
To load and use the saved model for making predictions:

python
Copy code
# Load the trained model
loaded_model = joblib.load('knn_iris_model.joblib')

# Example prediction
sample_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = loaded_model.predict(sample_data)
print("Predicted class:", prediction)