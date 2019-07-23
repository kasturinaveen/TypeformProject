from core.training import Training, Preprocessing, DataSource
from core.inference import Inference


class SubmissionRatioApp(object):
    def __init__(self, spark_context, sql_context):
        self.sql_context = sql_context
        self.spark_context = spark_context
        self.preprocess = Preprocessing()
        self.inference = Inference()

    def training_pipeline(self):
        """ Training Pipeline calls includes preprocessing, vectorzing and training
        """
        stages = list()
        data = DataSource().read_csv(self.sql_context)
        data = data.sample(False, 0.01, seed=20)
        train, test = data.randomSplit([0.8, 0.2])

        # stages.append(preprocess.preprocess)
        prep_data = self.preprocess.preprocess(train)
        vector_assemble = self.preprocess.vector_assemble(prep_data)
        stages.append(vector_assemble)
        trainer = Training()
        model = trainer.rf_train(prep_data, stages)
        prep_data = self.preprocess.preprocess(test)
        rmse = trainer.evaluate(prep_data)
        return rmse

    def inference_pipeline(self, data):
        """ Inference Pipeline calls includes preprocessing, inference
        """
        prep_data = self.preprocess.preprocess(data)
        prediction = self.inference.inference(prep_data)
        return prediction








