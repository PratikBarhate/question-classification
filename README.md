## question-classification
Classifier for the question classification dataset - [ http://cogcomp.org/Data/QA/QC/ ]

#### Execution
1. Go to the project directory.
2. We need to execute the command `./bin/qc.sh nlp` first.
3. Once the Natural Language Processing (NLP) is done for computing annotated natural language property we can train one of the models.
4. To train a model run command `./bin/qc.sh train {ml_algo_model}`. e.g `./bin/qc.sh train svm`
5. To test a model run command `./bin/qc.sh test {ml_algo_model}`.

##### Machine learning algorithms implemented - {ml_algo_model}
1. `svm` = Support Vector Machine
2. `lr` = Logistic Regression
3. `linear_svm` = Linear Support Vector Classifier (Machine)

#### Dependencies used

1. python - v3.6.3
2. configobj - v5.0.6
3. spaCy - v2.0.9 (with "en_core_web_lg" english model)
4. sner - v0.2.3
5. scipy - v1.0.0
6. scikit-learn - v0.19.1

###### *NOTE:*
###### *1. Tab = 4 spaces*
###### *2. command `python` should point to the installation following the above mentioned dependencies*
###### *3. Or you can change the command in the shell script `qc.sh` to the suitable python command.*
###### *python -m {operation} -> python3 -m {operation}*
