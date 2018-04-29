#### Natural Language processing

All the questions (rows) are processed together from both the training as well as test data, when you execute 
`./bin/qc.sh nlp`. The processed information from spaCy lib is stored as list of [Doc](https://spacy.io/api/doc) 
objects, where each element in the list represents the question (row) from the dataset of same index.
Name Entity Recognition (NER) tag is generated using StanfordNER and stored as list of tags, again where each element in
the list represents the same question (row) from the dataset.

All of the data is stored in a directory `{project_root_directory}/common_data/nlp`.

#### Data Preparation

1. To convert the text features, computed using the NLP lib (spaCy and stanfordNER), to numeric features (data) 
CountVectorizer from scikit-learn is used. The vectorizer used while training should be used while testing, 
hence the fitted vectorizers are stored in their respective {ml_algo} directories. Each individual model will have its own
vectorizer for each of the features used in 'feature_stack'.
2. While testing the {ml_algo} the same vectorizers are used to transform the test data.
3. All the various features are stacked together in `qc.dataprep.feature_stack`. It should be same as used while training.

#### Machine Learning (ML) - train phase

1. There are two set of labels in the data - coarse and fine. Structure of dataset is 
`coarseClass:fineClass question sentence` e.g. `DESC:def What is an atom ?`. Hence, here coarseClass = DESC and 
fineClass = def. Detailed taxonomy can be found at - [Taxonomy](http://cogcomp.org/Data/QA/QC/definition.html). 
2. There are total 7 ML models trained when you execute `./bin/qc.sh train {ml_algo}` command. The API to be used / type
of ML class to be loaded is decided from the {ml_algo} parameter and each of the 7 models are of the same class.
3. One model, namely `coarse`, is for predicting the coarseClass. To train this model all the training data 
(questions/rows) is used. With coarseClass as the target/dependent variable. fineClass is ignored.
4. Other six are to predict the fine class of the respective coarseClass (`abbr` | `desc` | `enty` | `hum` | `loc` | `num`).
The training data (questions/rows) which have the specific coarseClass is used for training the particular model.
With fineClass as the target/dependent variable. e.g `DESC:def What is an atom ?` and 
`DESC:reason Why does the moon turn orange ?`, are used to train `desc` model, with targets being `def` and `reason` 
respectively for the mentioned example questions.

*NLP is done once for all*

##### ->| Steps

1. DataPrep for each - `coarse` |`abbr` | `desc` | `enty` | `hum` | `loc` | `num`
2. Train ML models - `coarse` |`abbr` | `desc` | `enty` | `hum` | `loc` | `num` (using data from DataPrep step of the respective model)

#### Machine Learning (ML) - test phase

1. All of the test data is transformed (vectorized) using vectorizers of all of the 7 models.
2. Test data is traversed as per the question (row level data) in test data and transformed data using indexes.
3. First the prediction from the `coarse` model is used to predict the coarseClass.
4. Depending on the output from the above step, one of the model among - `abbr` | `desc` | `enty` | `hum` | `loc` | `num`,
is used to predict the fineClass.

##### ->| Steps

1. DataPrep of test data using all of the vectorizers - `coarse` |`abbr` | `desc` | `enty` | `hum` | `loc` | `num`.
You can observe a change in `qc.dataprep.text_feature.text_ft_arr` that while test data is processed only vectorizers
are changed and all the data is transformed using various vector models.  
2. Predict output using `coarse` ML model, using DataPrep data from `coarse` vectorizer output.
3. Similarly, as step 2, one of the model among - `abbr` | `desc` | `enty` | `hum` | `loc` | `num`, is used to predict
fineClass depending on the output from the step 2. e.g if the prediction from step 2 is "HUM" then `hum` mode ML model, 
using DataPrep data from `hum` vectorizer output.