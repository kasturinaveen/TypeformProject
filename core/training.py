import numpy as np
from pyspark.sql import functions
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor


class DataSource(object):
    def read_csv(self, sql_context):
        data = sql_context.read.csv('../data/data.csv')
        return data


class Preprocessing(object):
    """  This class handles data orientation, feature extraction, vector assembler
    """
    def __init__(self):
        self.assembler = None

    def preprocess(self, data):
        data = data.withColumn("_c0", functions.expr("substring(_c0, 2, length(_c0)-1)"))
        data = data.withColumn("_c3", functions.expr("substring(_c3, 1, length(_c3)-1)"))
        data = data.withColumnRenamed("_c0", "form_id") \
            .withColumnRenamed("_c1", "views") \
            .withColumnRenamed("_c2", "submissions") \
            .withColumnRenamed("_c3", "features")

        data = data.select('form_id', 'views', 'submissions', functions.split('features', '-').alias('features'))
        df_sizes = data.select(functions.size('features').alias('features'))
        df_max = df_sizes.agg(functions.max('features'))
        nb_columns = df_max.collect()[0][0]
        data = data.select('form_id', 'views', 'submissions', *[data['features'][i] for i in range(nb_columns)])

        data = data.select(*(functions.col(column).cast("float").alias(column) for column in data.columns))
        data = data.withColumn('form_id', functions.col('form_id').cast('int'))
        data = data.withColumn('views', functions.col('views').cast('int'))
        data = data.withColumn('submissions', functions.col('submissions').cast('int'))
        data = data.withColumn("submission_ratio", functions.col("submissions") / functions.col("views"))
        return data

    def vector_assemble(self, data):
        feature_columns = data.columns[3:-1]
        self.assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        return self.assembler

    def assembler_transform(self,data):
        return self.assembler.transform(data)


class Training(object):
    def __init__(self):
        self.model = None
        self.evaluator = None

    def rf_train(self, data, stages):
        """  Random forest training using Grid Search CV
        """
        rf = RandomForestRegressor(featuresCol='features', labelCol="submission_ratio")
        stages.append(rf)
        pipeline = Pipeline(stages=stages)

        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [int(x) for x in np.linspace(start=10, stop=50, num=3)]) \
            .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start=5, stop=25, num=3)]) \
            .build()

        self.evaluator = RegressionEvaluator(
            predictionCol='prediction',
            labelCol='submission_ratio',
            metricName='rmse',
        )
        cross_val = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=self.evaluator, numFolds=3)
        self.model = cross_val.fit(data)
        pip_model = self.model.bestModel
        pip_model.save("../data/model")

    def evaluate(self, data):
        """  Regression model evaluation
        """
        prediction = self.model.transform(data)
        rmse = self.evaluator.evaluate(prediction)
        return rmse










