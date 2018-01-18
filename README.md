## Deploying a Machine Learning Model via a RESTful API

### Python version
For best results use Python 3.x

### Sample cURL command
`curl -XPOST localhost:5000/predict -H 'Content-Type: application/json' -d '
{ 
"loan_amnt": 10000,
"term": "36 months",
"emp_length": 0,
"home_ownership": "OTHER",
"annual_inc": 30000,
"verification_status": "not verified",
"purpose": "car",
"addr_state": "AK",
"dti": 10,
"delinq_2yrs": 40,
"revol_util": 100.0,
"total_acc": 1,
"longest_credit_length": 0
}'`


### References
- https://github.com/h2oai/app-consumer-loan
- https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/building-a-smarter-application/BuildingASmarterApplication.pdf
- https://www.lendingclub.com/info/download-data.action

