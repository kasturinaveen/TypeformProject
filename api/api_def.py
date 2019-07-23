from flask import Blueprint
import json
import logging
from core.app import SubmissionRatioApp
from flask import Flask, request

main = Blueprint('main', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@main.route("/train", methods=["GET"])
def train():
    """ API end point for training
    """
    logger.debug("Training Started")
    rmse = submission_ratio_app.training_pipeline()
    logger.debug("Training Ended")
    return json.dumps(rmse)


@main.route("/inference", methods=["POST"])
def inference():
    """ API end point for inference
    """
    payload = request.get_json()
    data = payload['data']
    logger.debug("Inference")
    prediction = submission_ratio_app.inference_pipeline(data)
    return prediction.to_json(orient='records')


def create_app(spark_context, sql_context):
    """ Creating Flask app
    """
    global submission_ratio_app
    submission_ratio_app = SubmissionRatioApp(spark_context, sql_context)
    app = Flask(__name__)
    app.register_blueprint(main)
    return app