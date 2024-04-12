import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference

X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = np.array([0, 1, 0])
X_test = np.array([[2, 3, 4], [5, 6, 7]])

def test_train_model():
    """
    Test if the model training function returns a trained model.
    """
    # Train the model
    model = train_model(X_train, y_train)
    
    # Check if the trained model is not None & is a Random Forest
    assert model is not None
    assert isinstance(model, RandomForestClassifier), "Returned object is not a model of type RandomForestClassifier"


def test_inference():
    """
    Test if the inference function returns valid predictions.
    """
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    predictions = inference(model, X_test)
    
    # Check if predictions are not empty and have the correct shape
    assert len(predictions) > 0
    assert predictions.shape == (len(X_test),)


def test_dataset_properties():
    """
    Test if the training and test datasets have the expected size and data type.
    """
    # Define expected sizes and data types
    expected_train_size = (len(X_train), len(X_train[0]))
    expected_test_size = (len(X_test), len(X_test[0]))
    expected_data_type = np.ndarray
    
    # Check the size and data type of the training dataset
    assert X_train.shape == expected_train_size, "Training dataset size does not match"
    assert isinstance(X_train, expected_data_type), "Training dataset is not of expected data type"
    
    # Check the size and data type of the test dataset
    assert X_test.shape == expected_test_size, "Test dataset size does not match"
    assert isinstance(X_test, expected_data_type), "Test dataset is not of expected data type"