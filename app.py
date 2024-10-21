from flask import Flask, request, jsonify
from.src.pipelines.prediction_pipeline import CustomData
from src.exception import CustomException

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Get the JSON data from the request
        data = request.json
        
        # Extract values safely from the incoming JSON
        custom_data = CustomData(
            age=data.get("Age"),
            sex=data.get("Sex"),
            chest_pain_type=data.get("ChestPainType"),
            resting_bp=data.get("RestingBP"),
            cholesterol=data.get("Cholesterol"),
            fasting_bs=data.get("FastingBS"),
            resting_ecg=data.get("RestingECG"),
            max_hr=data.get("MaxHR"),
            oldpeak=data.get("Oldpeak"),
            exercise_angina=data.get("ExerciseAngina"),
            st_slope=data.get("ST_Slope")
        )

        # Convert to DataFrame
        fe = custom_data.get_data_as_dataframe()
        
        # Create a PredictPipeline instance and make a prediction
        prediction = PredictPipeline().predict(input_df) # type: ignore
        
        return jsonify({"prediction": prediction.tolist()})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except CustomException as ce:
        return jsonify({"error": str(ce)}), 500
    except Exception as e:
        return jsonify({"error": "An error occurred: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
