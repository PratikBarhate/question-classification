from qc.utils.file_ops import read_file
from qc.dataprep.feature_stack import get_ft_obj
from sklearn import svm


def execute(project_root_path: str):
    """
    :argument
        :param project_root_path: Absolute Path of the project
    :return:
        None
    """


def train_one_node(project_root_path: str, cat_type: str):
    """
    :argument:
        :param project_root_path: Absolute path of the root directory of the project.
        :param cat_type: Type of categorical class `coarse` or any of the 6 main classes.
                                        (`abbr` | `desc` | `enty` | `hum` | `loc` | `num`)
    :return:
        boolean_flag: True for successful operation.
        model: trained SVC model
    """
    x_ft = get_ft_obj("training", project_root_path, "svm", "coarse")
    rf, coarse_c = read_file("{0}_classes_training".format(cat_type), project_root_path)
    y_lb = [len(c) for c in coarse_c]
    machine = svm.SVC(kernel="rbf", C=1.0, gamma=0.0005)
    print("\nft -> {0}".format(len(x_ft)))
    print("\nlable -> {0}".format(len(y_lb)))
    print(len(x_ft) == len(y_lb))
    print("training started....-_-...")
    model = machine.fit(x_ft, y_lb)
    print("training complete !!")


train_one_node("/Users/tkmahxk/Pratik/Study/Projects/question-classification", "coarse")
