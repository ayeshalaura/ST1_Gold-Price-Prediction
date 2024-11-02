# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


from flask import Flask, render_template, request
import joblib
import os
import json
import pandas as pd
import glob
import shutil

app = Flask(
    __name__,
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../static')),
    static_url_path='/static'
)
app.secret_key = 'REDACTED'

def get_latest_model():
    """
    Retrieves the most recently saved model from the deployed models directory.
    """
    model_dir = os.path.join('models', 'deployed')
    model_files = glob.glob(os.path.join(model_dir, '*.joblib'))
    if not model_files:
        raise FileNotFoundError("No deployed models found in 'models/deployed'.")
    latest_model_file = max(model_files, key=os.path.getmtime)
    model = joblib.load(latest_model_file)
    model_filename = os.path.splitext(os.path.basename(latest_model_file))[0]
    # Remove '_deployed' suffix from model name to match image filenames
    if model_filename.endswith('_deployed'):
        model_name = model_filename[:-9]
    else:
        model_name = model_filename
    return model, model_name

def load_demo_data():
    """
    Loads the demo data from the JSON file, excluding the 'Adj Close' value.
    """
    demo_file = os.path.join('data', 'input', 'demo', 'last_rows.json')
    with open(demo_file, 'r') as f:
        data = json.load(f)
    actual_adj_close = [row.pop('Adj Close', None) for row in data]
    return data, actual_adj_close

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Retrieve existing predictions from the form
            predictions_json = request.form.get('predictions_json', '[]')
            predictions = json.loads(predictions_json)
            
            model, model_name = get_latest_model()
            data, actual_adj_close = load_demo_data()

            # Determine the next prediction index
            prediction_count = len(predictions)
            if prediction_count >= len(data):
                return "All available predictions have been made."

            input_data = data[prediction_count]
            actual_value = actual_adj_close[prediction_count]

            df = pd.DataFrame([input_data])
            scaler_path = os.path.join('models', 'scaler.joblib')
            scaler = joblib.load(scaler_path)
            df_scaled = scaler.transform(df)
            df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)

            predicted_adj_close = model.predict(df_scaled)[0]

            # Append the new prediction
            predictions.append((predicted_adj_close, actual_value))

            # Path to the visualization image
            image_filename = f'actual_vs_predicted_{model_name}.png'
            image_source_path = os.path.join('reports', 'visualizations', image_filename)
            static_image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../static', 'visualizations'))
            os.makedirs(static_image_dir, exist_ok=True)
            static_image_path = os.path.join(static_image_dir, image_filename)

            # Ensure the image exists in the static directory
            if os.path.exists(image_source_path):
                if not os.path.exists(static_image_path):
                    shutil.copy(image_source_path, static_image_path)
            else:
                image_filename = None  # Handle missing image gracefully

            return render_template(
                'index.html',
                predictions=predictions,
                model_name=model_name,
                image_filename=image_filename
            )
        except Exception as e:
            return f"An error occurred: {str(e)}"
    # For GET request, initialize empty predictions
    return render_template('index.html', predictions=[], model_name=None, image_filename=None)

if __name__ == '__main__':
    app.run(debug=True)
