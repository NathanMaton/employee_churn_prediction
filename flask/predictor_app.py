import flask
from flask import request
from predictor_api import make_prediction, feature_names

# Initialize the app

app = flask.Flask(__name__)


# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!
@app.route("/")
def hello():
    return "It's alive!!!"


feature_names2 = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'IT', 'RandD', 'accounting', 'hr',
       'management', 'marketing', 'product_mng', 'sales', 'support',
       'technical', 'high', 'low', 'medium']

@app.route("/predict", methods=["POST", "GET"])
def predict():
    return flask.render_template('predictor2.html',
                                 feature_names=feature_names2)


@app.route("/results", methods=["POST", "GET"])
def cv_results():
    x_input, predictions = make_prediction(request.args)
    return flask.render_template('results.html',x_input=x_input,
                                 feature_names=feature_names,
                                 prediction=predictions)


@app.route("/slider", methods=["POST", "GET"])
def slider():
    return flask.render_template('sliders.html')
# Start the server, continuously listen to requests.
# We'll have a running web app!

# For local development:
app.run(debug=True)

# For public web serving:
# app.run(host='0.0.0.0')
