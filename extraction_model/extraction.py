import os
import sys
from extraction_model.config import config
from extraction_model.pipeline_manager import create_spark_session
import logging
from extraction_model.log_manager import init_log

import sparknlp
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline

from pyspark.ml import Pipeline
from pyspark.sql import functions as F  
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, StrictStr
import uvicorn
import warnings
warnings.filterwarnings("ignore")

logger= init_log("NER pipeline", logging.INFO)


app= FastAPI()

class NerPipeline(BaseModel):
    text: str # type: ignore  

@app.get("/")
def index():
    return {"message": "Welcome to the Named Entity Recognition API"}


@app.get("/extractor")
def entity_extraction():
    """
    Extract entities from the example text
    """

    text= config.INPUT_TEXT
    
    logger.info("Received request body: {}".format(text))

    # creating spark session
    spark = create_spark_session()

    ner_piepline= PretrainedPipeline.from_disk(os.path.join(config.NER_PIPELINE_DIR, config.NER_PIPELINE_NAME))
    logger.info("NER pipeline is loaded successfully")

    # display the result as a list
    result_list= ner_piepline.annotate(text)["ner_chunk"]
    logger.info("Entities are extracted successfully")

    return {"extracted entities from the input text": result_list}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=0000)