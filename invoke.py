import requests
import json

scoring_uri = "http://ac113154-2e6f-499c-9be7-09bd5fba4d3a.northuae.azurecontainer.io/score"

# JSON stringify
data = {
    "data":
    [
        {
            'id': "0",
            'bmi': "0",
            'age': "80",
            'gender': "1",
            'work_type': "0",
            'hypertension': "1",
            'heart_disease': "0",
            'Residence_type': "0",
            'smoking_status': "0",
            'ever_married': False,
            'avg_glucose_level': "0",
        },
      ]
    }

data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(data)

headers = {'Content-Type': 'application/json'}

# Make the request (no authentication)
resp = requests.post(scoring_uri, data, headers=headers)
print(resp.json())