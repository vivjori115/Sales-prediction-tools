from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the pre-trained model (assumed to be a regression model)
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """
    Render the homepage (index.html) where the user can input data for prediction.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    """
    Handles the form submission, makes a prediction, and displays the result.
    """
    try:
        # Get the input values from the form
        TV = float(request.form.get('TV'))           # TV Advertising Budget
        Radio = int(request.form.get('Radio'))      # Radio Advertising Budget
        Newspaper = int(request.form.get('Newspaper'))  # Newspaper Advertising Budget

        # Make prediction using the model
        input_data = np.array([TV, Radio, Newspaper]).reshape(1, 3)
        result = model.predict(input_data)

        # Return the prediction result back to the user on the same page
        return render_template('index.html', result=result[0])

    except ValueError:
        # Handle case when user inputs invalid values
        return render_template('index.html', result="Invalid input. Please enter valid numbers.")

if __name__ == '__main__':
    # Run the app on all available network interfaces (host='0.0.0.0') and port 8080
    app.run(host='0.0.0.0', port=8080)
