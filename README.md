## question-classification
Classifier for the question classification [dataset](http://cogcomp.org/Data/QA/QC/).

1. _Results from the empirical tests carried out are in `{project_directory}/documentation/Results.md`_
2. _More details about the execution/logic is available in `{project_directory}/documentation/Execution_Details.md`_

#### Dependencies used

1. python - v3.6.3
2. configobj - v5.0.6
3. spaCy - v2.0.9 (with "en_core_web_lg" english model)
4. sner - v0.2.3
5. scipy - v1.0.0
6. scikit-learn - v0.19.1

#### Execution

_Check your systems' text encoding scheme. It is set to `text_file_encoding = "utf8"`, can be changed in `qc.utils.file_ops.py`._

1. Go to the project directory.
2. We need to execute the command `./bin/qc.sh nlp` first.
3. Once the Natural Language Processing (NLP) is done for computing annotated natural language property we can train 
   one of the models.
4. To train a model run command `./bin/qc.sh train {ml_algo_model}`. e.g `./bin/qc.sh train svm`
5. To test a model run command `./bin/qc.sh test {ml_algo_model}`.

##### Machine learning algorithms implemented - {ml_algo_model}

1. `svm` = Support Vector Machine
2. `lr` = Logistic Regression
3. `linear_svm` = Linear Support Vector Classifier (Machine)

##### To clean the outputs

1. `./bin/cleanup.sh nlp` - This will delete all the NLP related data.
2. `./bin/cleanup.sh models` - This will delete all the pre-trained models.
3. `./bin/cleanup.sh {ml_algo_model}` - This will delete the specific ML model which was pre-trained.
4. `./bin/cleanup.sh all` - This will delete all the computed data.

#### Experimental Code

1. The method to convert text data to ML features can be modified in function `qc.dataprep.text_features.get_vect`.
2. The feature stack (what all data is to be feed to ML algorithm) can be modified/transformed/generated 
   in file `qc.dataprep.feature_stack`
   
   _These (point 1, 2) changes are used whenever you execute training process again. 
   There is no need to execute `nlp` step again._
   
3. Machine learning algorithms can be added in function `qc.ml.train.train_one_node`. (Parameter tuning too can be done)
   e.g In the experimental part of the code add extra `elif` statement
   
   ```
   elif == {your_model_name}:
       machine = {Initialize the algorithm you want to use}
   ```
   
   ```
   elif ml_algo == "lr_lsvm":
       if cat_type == "coarse":
           machine = linear_model.LogisticRegression(solver="newton-cg")
       else:
           machine = svm.LinearSVC() 
   ```
   
   While executing using the shell command `./bin/qc.sh train lr_lsvm`, and this command will use the model defined by you. 
   `lr_lsvm` is `{your_model_name}`. In the example we have defined to use LogisticRegression
   for coarse class prediction and LinearSVC for fine class predictions (all of the fine class predictions). 

###### *NOTE:*
###### *1. Tab = 4 spaces*
###### *2. command `python` should point to the installation following the above mentioned dependencies*
###### *3. Or you can change the command in the shell script `qc.sh` to the suitable python command.*
###### *python -m {operation} ---> python3 -m {operation}*

#### License

[MIT License](https://github.com/Pratik-Barhate/question-classification/blob/master/LICENSE)

#### Credits

This project has been inspired from one of the problem we tried to solve - understanding the question for our QA bot. 
In a project named `Invoker`, I did work with [Akash Pateria](https://github.com/Akash-Pateria), we worked together 
in the final year graduate project. We did use python - v2.7 and [practNLPtools](https://github.com/biplab-iitb/practNLPTools) 
for our tasks in the project 'Invoker'.

This project aims at exploring more options to process Natural Language (English), test with various combinations of 
features and improve the accuracy.