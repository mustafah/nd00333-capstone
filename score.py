import json
import pandas as pd
import joblib
from azureml.core.model import Model
import os



def init():
    global model

    try:
        # logger.info("Loading model from path.")

        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
        # logger.info('****')
        # logger.info(model_path)
        model = joblib.load(model_path)

        # logger.info("Loading successful.")
    except Exception as e:
        # logger.info(e)
        # logging_utilities.log_traceback(e, logger)
        raise


def run(data):
    try:
        test_data = json.loads(data)
        data_frame = pd.DataFrame(test_data['data'])
        result = model.predict(data_frame)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        error = str(e)
        # logger.info(e)
        # logging_utilities.log_traceback(e, logger)
        return error
