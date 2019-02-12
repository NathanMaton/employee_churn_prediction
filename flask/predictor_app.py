import flask
from flask import request, jsonify
from predictor_api import make_prediction, feature_names

# Initialize the app

app = flask.Flask(__name__)


# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!
@app.route("/")
def hello():
    return flask.render_template('home.html')

@app.route("/contact")
def contact():
    return flask.render_template('contact.html')



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

    #pippo =  request.form.getlist('name[]')
    x_input, predictions = make_prediction(request.form.getlist())
    return flask.render_template('results.html',x_input=x_input,
                                 feature_names=feature_names,
                                 prediction=predictions)

@app.route("/predict_api", methods=["POST"])
def get_api_response():
    # This function will throw an error if we are missing one of
    # the features it expects. Any status code that is not 200 - 299
    # is flagged as an error
    try:
        response = make_prediction(request.json)
        #print ('inside route endpoint')
        #print (response)
        status = 200
    except KeyError:
        missing = [f for f in feature_names if f not in request.json]
        response = {
            'status': 'error',
            'msg': f'not all required feature names ({feature_names}) present. Missing {missing}'
        }
        status = 300
    return response, status



@app.route("/slider", methods=["POST", "GET"])
def slider():
    return flask.render_template('sliders.html')
# Start the server, continuously listen to requests.
# We'll have a running web app!

# For local development:
app.run(debug=True)

# For public web serving:
# app.run(host='0.0.0.0')
