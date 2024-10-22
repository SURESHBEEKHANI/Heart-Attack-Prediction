from flask import Flask, render_template, request, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Get the form data from the request
        data = request.form
        
        # Extract values safely from the incoming form data
        custom_data = CustomData(
            age=data.get("age"),
            sex=data.get("sex"),
            chest_pain_type=data.get("chestPainType"),
            resting_bp=data.get("restingBP"),
            cholesterol=data.get("cholesterol"),
            fasting_bs=data.get("fastingBS"),
            resting_ecg=data.get("restingECG"),
            max_hr=data.get("maxHR"),
            oldpeak=data.get("oldpeak"),
            exercise_angina=data.get("exerciseAngina"),
            st_slope=data.get("stSlope")
        )

        # Convert to DataFrame
        input_df = custom_data.get_data_as_dataframe()

        # Create a PredictPipeline instance and make a prediction
        prediction_pipeline = PredictPipeline()
        prediction = prediction_pipeline.predict(input_df)

        # Condition to check if prediction equals 1
        if prediction[0] == 1:
            result_message = "You are at moderate risk of experiencing a heart attack"
        else:
            result_message = "There are no immediate risk factors for a heart attack"

        # Pass the result message back to the template for display
        return render_template('index.html', results=result_message)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except CustomException as ce:
        return jsonify({"error": str(ce)}), 500
    except Exception as e:
        return jsonify({"error": "An error occurred: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
