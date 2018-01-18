from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/index')
def index():
    return 'App Consumer Loan'


@app.route('/predict', methods=['POST'])
def predict():
    # Load data
    data_json = request.json
    data = pd.DataFrame(data_json, index=[0], columns=columns)

    # Preprocess
    # data['term'] = data.term.apply(lambda x: int(x.split(' ')[0])).replace([36, 60], [0, 1])
    # data['verification_status'] = data.verification_status.replace(['verified', 'not verified'], [1, 0])

    # Decode Labels
    cat_cols = ['term', 'home_ownership', 'purpose', 'addr_state', 'verification_status']
    cat_data = data[cat_cols]
    data = data.drop(cat_cols, axis=1)  # Isolate cat variables for preprocessing
    cat_data_encoded = np.empty(data.shape[0])
    # LabelEncode the cat variables
    for i, col in enumerate(cat_cols):
        encoded = label_encoders[i].transform(cat_data[col])
        cat_data_encoded = np.column_stack((cat_data_encoded, encoded))

    cat_data_encoded = np.delete(cat_data_encoded, obj=0, axis=1)
    # OneHotEncode
    cat_data_encoded = onehot_encoder.transform(cat_data_encoded).toarray()

    # Finally recombine
    x = np.column_stack((data.values, cat_data_encoded))

    result = loan_default_model.predict(x).tolist()[0]
    return jsonify(
        loan_default_prediction=result
    )

    # loan_amnt = float(request.args.get('loan_amnt'))
    # term = int(request.args.get('term').split[' '][0])
    # emp_length = int(request.args.get('emp_length'))
    # if request.args.get('verification_status') == 'verified':
    #     verification_status = 1
    # else:
    #     verification_status = 0
    #
    # # Preprocess with label encoder
    # home_ownership = label_encoders[0].tranform([request.args.get('home_ownership')])
    # purpose = label_encoders[1].tranform([request.args.get('purpose')])
    # addr_state = label_encoders[2].tranform([request.args.get('addr_state')])
    # # Preprocess with label encoder
    #
    # annual_inc = float(request.args.get('annual_inc'))
    # dti = float(request.args.get('dti'))
    # delinq_2yrs = int(request.args.get('delinq_2yrs'))
    # revol_util = int(request.args.get('revol_util'))
    # total_acc = int(request.args.get('total_acc'))
    # longest_credit_length = float(request.args.get('longest_credit_length'))


if __name__ == '__main__':
    columns = joblib.load('../results/columns.pkl')
    label_encoders = joblib.load('../results/label_encoders.pkl')
    onehot_encoder = joblib.load('../results/onehot_encoder.pkl')
    loan_default_model = joblib.load('../results/final_model.pkl')
    app.run(debug=True)
