import pathlib
import os
import extraction_model

# Path to the root directory of the project
ROOT_DIR = pathlib.Path(extraction_model.__file__).resolve().parent

# Path to the pretrained models
PRETRAINED_MODEL_DIR = os.path.join(ROOT_DIR, 'pretrained_models')

# Pretrained models
SPELL_CHECKER_MODEL = "spellcheck_dl_en_3.4.1_3.0_1648457196011"
NER_MODEL = "onto_100_en_2.4.0_2.4_1579729071672"
EMBEDDINGS_MODEL= "glove_100d_en_2.4.0_2.4_1579690104032"

#Path to the saved pipeline
NER_PIPELINE_NAME= "ner_pipeline"
NER_PIPELINE_DIR = os.path.join(ROOT_DIR, 'saved_ner_pipeline')

# Path to the sparknlp jar
SPARKNLP_JAR_DIR = os.path.join(ROOT_DIR, 'sparknlp_jar')
SPARKNLP_JAR_NAME = "spark-nlp-assembly-5.3.2.jar"

# example text
INPUT_TEXT = "John Lennon is a legend and he was born in Liverpool. He was a member of The Beatles."

