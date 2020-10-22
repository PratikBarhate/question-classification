from spacy.tokens.doc import Doc

from qc.dataprep.feature_stack import get_ft_obj
from qc.nlp.proc_coarse import com_annotations_param
from qc.utils.file_ops import read_obj
from flask import Flask, request, jsonify

coarse_model = None
abbr_model = None
desc_model = None
enty_model = None
hum_model = None
loc_model = None
num_model = None


def load_models(rp: str, ml_algo: str):
    global coarse_model, abbr_model, desc_model, enty_model, hum_model, loc_model, num_model
    #
    """
    Load all the trained models

    :argument
        :param rp: Absolute path of the root directory of the project.
        :param ml_algo: The type of machine learning models to be used. (svm | lr | linear_svm)
    """
    crf, coarse_model = read_obj("coarse_model", rp + "/{0}".format(ml_algo).format(ml_algo))
    arf, abbr_model = read_obj("abbr_model", rp + "/{0}".format(ml_algo))
    drf, desc_model = read_obj("desc_model", rp + "/{0}".format(ml_algo))
    erf, enty_model = read_obj("enty_model", rp + "/{0}".format(ml_algo))
    hrf, hum_model = read_obj("hum_model", rp + "/{0}".format(ml_algo))
    lrf, loc_model = read_obj("loc_model", rp + "/{0}".format(ml_algo))
    nrf, num_model = read_obj("num_model", rp + "/{0}".format(ml_algo))

    if not crf:
        print("- Error in reading coarse {0} model".format(ml_algo))
    if not arf:
        print("- Error in reading abbr {0} model".format(ml_algo))
    if not drf:
        print("- Error in reading desc {0} model".format(ml_algo))
    if not erf:
        print("- Error in reading enty {0} model".format(ml_algo))
    if not hrf:
        print("- Error in reading hum {0} model".format(ml_algo))
    if not lrf:
        print("- Error in reading loc {0} model".format(ml_algo))
    if not nrf:
        print("- Error in reading num {0} model".format(ml_algo))

    if crf and arf and drf and erf and hrf and lrf and nrf:
        print("- Loading the {0} models complete".format(ml_algo))
    else:
        print("- Error in loading the pre-trained model")
        exit(-10)


def nlp_process_question(question: str):
    return com_annotations_param(question)[1]


def get_predictions(question_doc: Doc, rp: str, ml_algo: str):
    """
    Gets data in the form of sparse matrix from `qc.dataprep.feature_stack` module
    which is ready for use in a machine learning model. Using the test data in `question-classification/dataset`
    tests [[ml_algo]] model which is pre-trained and saved as serialized pickle files.

    :argument
        :param question_doc The question to be classified given as tagged Doc from NLP process.
        :param rp: Absolute path of the root directory of the project.
        :param ml_algo: The type of machine learning models to be used. (svm | lr | linear_svm)
    :return:
        pred: Question class prediction.
    """

    c_ft = get_ft_obj("api", rp, ml_algo, "coarse", [question_doc]).tocsr()
    a_ft = get_ft_obj("api", rp, ml_algo, "abbr", [question_doc]).tocsr()
    d_ft = get_ft_obj("api", rp, ml_algo, "desc", [question_doc]).tocsr()
    e_ft = get_ft_obj("api", rp, ml_algo, "enty", [question_doc]).tocsr()
    h_ft = get_ft_obj("api", rp, ml_algo, "hum", [question_doc]).tocsr()
    l_ft = get_ft_obj("api", rp, ml_algo, "loc", [question_doc]).tocsr()
    n_ft = get_ft_obj("api", rp, ml_algo, "num", [question_doc]).tocsr()
    print("- DataPrep done.")

    c = coarse_model.predict(c_ft[0])[0]

    if c == "ABBR":
        f = abbr_model.predict(a_ft[0])[0]
    elif c == "DESC":
        f = desc_model.predict(d_ft[0])[0]
    elif c == "ENTY":
        f = enty_model.predict(e_ft[0])[0]
    elif c == "HUM":
        f = hum_model.predict(h_ft[0])[0]
    elif c == "LOC":
        f = loc_model.predict(l_ft[0])[0]
    else:
        f = num_model.predict(n_ft[0])[0]

    print("- Predict done.")

    return [c, f]


def start(project_root_path: str, ml_algo: str):
    """
    Starts a simple web server with a single endpoint to classify given questions.

    :argument
        :param project_root_path: Absolute Path of the project
        :param ml_algo: The type of machine learning models to be used. (svm | lr | linear_svm)
    :return:
        None
    """

    load_models(project_root_path, ml_algo)
    app = Flask(__name__)

    @app.route('/classify', methods=['POST'])
    def classify_question():
        json = request.get_json()

        if json is None or 'question' not in json:
            return 'Request does not contain valid JSON with the question attribute!', 400

        question = json['question']
        question_doc = nlp_process_question(question)
        pred = get_predictions(question_doc, project_root_path, ml_algo)

        return jsonify({
            "coarse_class": pred[0],
            "fine_class": pred[1]
        })

    port = 5003
    app.run(host='0.0.0.0', port=port)
    print("Server started at port {0}".format(port))
