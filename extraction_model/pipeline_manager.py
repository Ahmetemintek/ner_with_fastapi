import sparknlp
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.sql import functions as F  

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import logging
from extraction_model.config import config
from extraction_model.log_manager import init_log

logger= init_log("NER pipeline", logging.INFO)


def create_spark_session():
    """
    Create a spark session
    """

    logger.info("spark session is initializing")

    sparknlp_jar_path = os.path.join(config.SPARKNLP_JAR_DIR, config.SPARKNLP_JAR_NAME)


    spark = SparkSession.builder \
        .appName("Spark NLP")\
        .master("local[*]")\
        .config("spark.driver.memory","16G")\
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.kryoserializer.buffer.max", "2000M")\
        .config("spark.jars", "{}".format(os.path.join(config.SPARKNLP_JAR_DIR, config.SPARKNLP_JAR_NAME)))\
        .getOrCreate()
    
    logger.info("spark session is created successfully")
    return spark

def create_ner_pipeline():
    """
    Create a sparknlp pipeline for named entity recognition
    """

    spark= create_spark_session()

    spell_path= os.path.join(config.PRETRAINED_MODEL_DIR, config.SPELL_CHECKER_MODEL)
    ner_path= os.path.join(config.PRETRAINED_MODEL_DIR, config.NER_MODEL)
    emb_path= os.path.join(config.PRETRAINED_MODEL_DIR, config.EMBEDDINGS_MODEL)

    logger.info("Loading pretrained models and creating the NER pipeline")

    documentAssembler= DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")      

    tokenizer= Tokenizer()\
            .setInputCols(["document"])\
            .setOutputCol("token")     

    spell_checker= ContextSpellCheckerModel.load(spell_path)\
            .setInputCols("token")\
            .setOutputCol("checked")       

    word_embedding= WordEmbeddingsModel.load(emb_path)\
            .setInputCols(["document", "checked"])\
            .setOutputCol("embeddings")    

    onto_ner = NerDLModel.load(ner_path) \
            .setInputCols(["document", "checked", "embeddings"]) \
            .setOutputCol("ner")     

    ner_converter= NerConverter()\
            .setInputCols(["document", "checked", "ner"])\
            .setOutputCol("ner_chunk")     

    nlp_pipeline= Pipeline(stages=[ 
                                   documentAssembler,
                                   tokenizer,
                                   spell_checker,
                                   word_embedding,
                                   onto_ner,
                                   ner_converter
    ])

    
    empty_df= spark.createDataFrame([[" "]]).toDF("text")
    pipelineModel= nlp_pipeline.fit(empty_df)

    logger.info("NER pipeline was created and fitted succesfully.")
    
    return pipelineModel


#saving the pipeline
def save_pipeline():
    """
    Save the pipeline to a specified path
    """

    pipeline= create_ner_pipeline()

    pipeline.write().overwrite().save(os.path.join(config.NER_PIPELINE_DIR, config.NER_PIPELINE_NAME))

    logger.info("Pipeline saved successfully!")


if __name__ == '__main__':
    save_pipeline()