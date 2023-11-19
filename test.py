from api.app import app
import pytest

def test_post_predict():
    test_client = app.test_client()

    # Test each digit
    for digit in range(10):
        sample_data = get_sample_for_digit(digit)  # Fetch the sample for the current digit
        response = test_client.post("/predict", json={"data": sample_data})

        # Check if the status code is 200
        assert response.status_code == 200

        # Assuming the response data is a byte string of the predicted digit
        # Convert the byte string to int for comparison
        predicted_digit = int(response.get_data().decode())
        
        # Check if the returned predicted digit is correct
        assert predicted_digit == digit

    # Additional test for the status code with a default sample
    default_sample = get_sample_for_digit(0)  # Or any other default sample
    response = test_client.post("/predict", json={"data": default_sample})
    assert response.status_code == 200
