## question-classification

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)    [![Codacy Badge](https://api.codacy.com/project/badge/Grade/4a93dde781c5421d9078f49687df8bf1)](https://www.codacy.com/app/Pratik-Barhate/question-classification)    [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/PratikBarhate/question-classification/blob/master/LICENSE)

Classifier for the question classification [dataset](http://cogcomp.org/Data/QA/QC/) (UIUC's CogComp QC Dataset).

1. Results from the empirical tests carried out, are in [results](documentation/Results.md) file.
All the results are for `coarse:fine`, combined prediction class out of the total 50 classes, if not stated otherwise.
2. More details about the execution/logic is available in [execution details](documentation/Execution_Details.md).
3. Diagrammatic representation of the data flow can be accessed [here](documentation/Data_Flow_diagram.pdf).

* The data-flow is different for Neural Network, its only a single `coarse` model predicting for 
all 50 different classes.

#### Install

Python 3.6.3 required. See requirements.txt for the list of other dependencies or use pip (see below).

Example Linux setup using pyenv to install an older Python version and venv for installing dependencies inside the project dir:
 
```
# Install and select Python 3.6.3
pyenv install -v 3.6.3
pyenv local 3.6.3

# Create a project specific virtual envirtonment for installing dependencies
python -m venv venv
source venv/bin/activate

# Update pip and install required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download the english langiage model
python -m spacy download en_core_web_lg
```

#### Execution

###### `bin` executable files are helpful only for Linux or macOS users. Microsoft users please execute the python modules by matching your requirement from the shell scripts.

_Check your systems' text encoding scheme. It is set to `text_file_encoding = "utf8"`, can be changed in `qc.utils.file_ops.py`._

1. Go to the project directory.
2. We need to execute the command `./bin/qc.sh nlp` first.
3. Once the Natural Language Processing (NLP) is done for computing annotated natural language property we can train
   one of the models.
4. To train a model execute command `./bin/qc.sh train {ml_algo_model}`. e.g `./bin/qc.sh train svm`
5. To test a model execute command `./bin/qc.sh test {ml_algo_model}`.

* All the trained models are saved inside a folder named - `${ml_algo_model}`, inside the project's root directory.

##### Machine learning algorithms implemented - {ml_algo_model}

1. `svm` = Support Vector Machine
2. `lr` = Logistic Regression
3. `linear_svm` = Linear Support Vector Classifier (Machine)
4. `nn` = Neural Network

##### To clean the outputs

1. `./bin/cleanup.sh nlp` - This will delete all the NLP related data.
2. `./bin/cleanup.sh all_models` - This will delete all the pre-trained models.
3. `./bin/cleanup.sh model ${ml_algo_model}` - This will delete the specific ML model which was pre-trained.
4. `./bin/cleanup.sh all` - This will delete all the computed data.

* `all_models` will not clean the additional model defined by you. It will only clean the models mentioned above.

#### Experimental Code

1. The method to convert text data to ML features can be modified in function `qc.dataprep.text_features.get_vect`. [code location](qc/dataprep/text_features.py)
2. The feature stack (what all data is to be feed to ML algorithm) can be modified/transformed/generated
   in file `qc.dataprep.feature_stack`. [code location](qc/dataprep/feature_stack.py)

   _These (point 1, 2) changes are used whenever you execute training process again.
   There is no need to execute `nlp` step again._

3. Machine learning algorithms can be added in function `qc.ml.train.train_one_node`. [code location](qc/ml/train.py)
(Parameter tuning too can be done). *e.g* In the experimental part of the code add extra `elif` statement

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

   While executing, use the shell command `./bin/qc.sh train lr_lsvm`, and this command will use the model defined by you.
   `lr_lsvm` is `{your_model_name}`. In the example we have defined to use LogisticRegression
   for coarse class prediction and LinearSVC for fine class predictions (all of the fine class predictions).

###### *NOTE:*
###### *1. Tab = 4 spaces*
###### *2. command `python` should point to the installation following the above mentioned dependencies*
###### *3. Or you can change the command in the shell script `qc.sh` to the suitable python command.*
###### *python -m {operation} ---> python3 -m {operation}*

#### License

[MIT](LICENSE)

#### Credits

This project has been inspired from one of the problem we tried to solve - understanding the question for our QA bot.
In a project named `Invoker`, I did work with [Akash Pateria](https://github.com/Akash-Pateria), we worked together
in the undergraduate capstone project. We did use python - v2.7, [practNLPtools](https://github.com/biplab-iitb/practNLPTools), 
and LinearSVC, as the ML algorithm, for our tasks in the project `Invoker`.

This project aims at exploring more options to process Natural Language (English), test with various combinations of
features and improve the accuracy.

#### References

[High-Performance Question Classification Using Semantic Features](https://nlp.stanford.edu/courses/cs224n/2010/reports/olalerew.pdf)
