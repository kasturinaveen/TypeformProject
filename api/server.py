import cherrypy
from paste.translogger import TransLogger
from api.api_def import create_app
from pyspark import SparkContext, SparkConf, SQLContext


def init_spark_context():
    conf = SparkConf().setAppName("submission_ratio_prediction-server")
    sc = SparkContext(conf=conf, pyFiles=['../core/inference.py', '../core/app.py', '../core/training.py'])
    sql_context = SQLContext(sc)
    return sc, sql_context


def run_server(app):
    app_logged = TransLogger(app)
    cherrypy.tree.graft(app_logged, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 5432,
        'server.socket_host': '0.0.0.0'
    })

    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == "__main__":
    sc, sql_context = init_spark_context()
    app = create_app(sc, sql_context)

    run_server(app)