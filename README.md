# Named Entity Recognition with FastAPI

This project implements a Named Entity Recognition (NER) over an example text using FastAPI. <br/>
In the named entity recognition pipeline, following components were used:
- Document assembler
- Tokenizer
- Pretrained spell checker [spellcheck_dl](https://sparknlp.org/2022/04/02/spellcheck_dl_en_2_4.html)
- Pretrained word embeddings [glove_100d](https://sparknlp.org/2020/01/22/glove_100d.html)
- Pretrained NER model [onto_100](https://sparknlp.org/2020/02/03/onto_100_en.html)
- NER converter to create NER chunks. 

The pipeline created in this project is able to detect following entity types from given text: <br/>

 `CARDINAL`, `EVENT`, `WORK_OF_ART`, `ORG`, `DATE`, `GPE`, `PERSON`, `PRODUCT`, `NORP`, `ORDINAL`, `MONEY`, `LOC`, `FAC`, `LAW`, `TIME`, `PERCENT`, `QUANTITY`, `LANGUAGE` 


## Files

- `extraction_model/`:
  - `config/`
    - `config.py`: Configuration settings for the NER model.
  - `pretrained_models/`: A folder containing pre-trained sparknlp models. 
    - `glove_100d_en_2.4.0_2.4_1579690104032`: Pretrained word embeddings model
    - `onto_100_en_2.4.0_2.4_1579729071672`: Pretrained NER model
    - `spellcheck_dl_en_3.4.1_3.0_1648457196011`: Pretrained spell checker model
  - `saved_ner_pipeline/`: A folder containing saved NER pipeline. 
    - `ner_pipeline`: Named Entity Recognition pipeline that is created with this project. 
  - `saprknlp_jar/`
    - `spark-nlp-assembly-5.3.2.jar`: Jar file for the sparknlp library. 
  - `extraction.py`: FastAPI application code containing the endpoints for NER.
  - `log_manager.py`: Module for initializing the logger.
  - `pipeline_manager.py`: Module for managing the Spark NLP pipeline.
- `requirements.txt`: List of Python dependencies required to run the project.
- `README.md`: This README file providing information about the project.

