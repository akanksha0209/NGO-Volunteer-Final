import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import os


# Load the trained model
clf = joblib.load('trained_lg_model.joblib')

# Create a Flask app
app = Flask(__name__)
CORS(app)


# Define the prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the user
    input_data = request.get_json()
    print(input_data)

    new_volunteer = []
    for key, value in input_data.items():
        new_volunteer.append(int(value))
    print(new_volunteer)
    # Make a prediction using the trained model
    # new_volunteer = input_data['new_volunteer']
    predicted_ngo = clf.predict([new_volunteer])

    # Return the predicted result as JSON
    return jsonify({'predicted_ngo': predicted_ngo.tolist()})


# Run the Flask app on port 3003
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=int(os.environ.get('PORT', 3003)))
