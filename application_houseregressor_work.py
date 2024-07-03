
from flask import Flask, render_template, request,jsonify
import pandas as pd
import pickle
app = Flask(__name__)

# Load models
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet', 
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM',
     'KNN','LGBM','XGBoost'
]
models = {name: pickle.load(open(f'{name}.pkl', 'rb')) for name in model_names}

# Load evaluation results
results_df = pd.read_csv(r"C:\Users\sandh\OneDrive\Desktop\python\july1st_housing regrssor_13 algorithims_fronand back end\13_algo_regressor_deploy\model_evaluation_results.csv")


@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    input_data = {
        'Avg. Area Income': float(request.form['Avg. Area Income']),
        'Avg. Area House Age': float(request.form['Avg. Area House Age']),
        'Avg. Area Number of Rooms': float(request.form['Avg. Area Number of Rooms']),
        'Avg. Area Number of Bedrooms': float(request.form['Avg. Area Number of Bedrooms']),
        'Area Population': float(request.form['Area Population'])
    }
    input_df = pd.DataFrame([input_data])
    
    
    
    if model_name in models:
        model = models[model_name]
        prediction = model.predict(input_df)[0]
        return render_template('results.html', prediction=prediction, model_name=model_name)
    else:
        return jsonify({'error': 'Model not found'}), 400

@app.route('/results')
def results():
    return render_template('model.html', tables=[results_df.to_html(classes='data')], titles=results_df.columns.values)

if __name__ == '__main__':
    app.run(host='192.168.0.22',port=8889)


#from flask import Flask

#app = Flask(__name__)

#@app.route('/')
#def index():
    #return 'welcome to FLASK WEBSITE CREATEION  first front end model '


#app.run(host='0.0.0.0', port=81)

#app.run(host = '192.168.0.22', port=8889)

