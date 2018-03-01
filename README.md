## question-classification
Classifier for the question classification dataset - [ http://cogcomp.org/Data/QA/QC/ ]

#### Execution
1. Go to the project directory.
2. We need to execute the command `./bin/qc.sh nlp` first.
3. Once the Natural Language Processing (NLP) is done for computing annotated natural language property we can train one of the models.
4. To train a model run command `./bin/qc.sh train {model_name}`

#### Dependencies used

1. python - v3.6.3
2. configobj - v5.0.6
3. spaCy - v2.0.9 (with "en_core_web_lg" english model)
4. sner - v0.2.3
5. scipy - v1.0.0
6. scikit-learn - v0.19.1

###### *NOTE : Tab = 4 spaces*
