from pyspark.ml import PipelineModel


class Inference(object):
    def __init__(self):
        self.prediction = None
        self.cv_model = PipelineModel.load('../data/model')

    def inference(self, data):
        """  Inference using built model loaded from file system
        """
        predictions = self.cv_model.transform(data)
        return predictions.select('form_id', 'prediction').toPandas()









