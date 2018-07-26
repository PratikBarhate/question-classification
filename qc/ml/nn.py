import datetime

import torch
import torch.nn as nn
import torch.nn.functional as torch_func
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import TensorDataset, DataLoader

from qc.dataprep.feature_stack import get_ft_obj
from qc.pre_processing.raw_processing import remove_endline_char
from qc.utils.file_ops import read_file, write_obj, read_obj, read_key

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model_str = "nn"
num_of_classes = 50

# --------------------------------------------Experimental - Execution Parameters---------------------------------------
epochs = 50
batch_size = 50
learning_rate = 1E-6


# Execution parameter tuning ends here.
# ----------------------------------------------------------------------------------------------------------------------


class NeuralNet(nn.Module):
    """
    Class defines the type of Neural Network its connections
    and activation function for each layer.
    """

    def __init__(self, in_layer: int, out_layer: int):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_layer, 100000)
        self.fc2 = nn.Linear(100000, 50000)
        self.fc3 = nn.ReLU()
        self.fc4 = nn.Linear(50000, out_layer)

    def forward(self, x):
        x = torch_func.tanh(self.fc1(x))
        x = torch_func.tanh(self.fc2(x))
        x = torch_func.tanh(self.fc3(x))
        x = self.fc4(x)
        return torch_func.log_softmax(x)


def get_data_loader(rp: str, data_type: str):
    """
    Reads the labels for the training data and converts it into a tensor
    of "(features, target)" for Neural Network using PyTorch.
    1. Loads the training classes, both coarse and fine and creates label for
    each row by "concatenating coarse_class :: fine_class".
    2. Converts the labels_numpy to labels_bin - Binarized form to be used in NN.
    3. Loads the features using the `get_ft_obj` function in numpy arrays.
    4. Get the number of features used.
    5. Converts label_numpy into PyTorch tensor - labels.
    6. Converts x_ft(features - independent variables) into PyTorch

    :argument
        :param rp: Absolute path of the root directory of the project.
        :param data_type: String either `training` or `test`.

    :return:
        feat_size: Number of features which are being used, so that we can keep the
        data_loader: Loader object containing train data and labels used to train the Neural Network.
    """
    labels_numpy = []
    crf, coarse = read_file("coarse_classes_{0}".format(data_type), rp)
    frf, fine = read_file("fine_classes_{0}".format(data_type), rp)
    c_lb = [remove_endline_char(c).strip() for c in coarse]
    f_lb = [remove_endline_char(f).strip() for f in fine]
    if not crf:
        print("Error in reading actual ({0}) coarse classes".format(data_type))
        exit(-11)
    if not frf:
        print("Error in reading actual ({0}) fine classes".format(data_type))
        exit(-11)
    for i in range(0, len(f_lb)):
        labels_numpy.append(c_lb[i] + " :: " + f_lb[i])
    mlb = MultiLabelBinarizer().fit(labels_numpy) if data_type == "training" \
        else read_obj("label_binarizer", rp + "/{0}".format(nn_model_str))[1]
    labels_bin = mlb.transform(labels_numpy)
    write_obj(mlb, "label_binarizer", rp + "/{0}".format(nn_model_str))
    print("- Labels loading into numpy done.")
    x_ft = get_ft_obj(data_type, rp, "{0}".format(nn_model_str), "coarse").toarray()
    feat_size = x_ft.shape[1]
    print("- Features loading into numpy done.")
    labels = torch.from_numpy(labels_bin)
    data = torch.from_numpy(x_ft)
    print("- Features and labels as tensors, done.")
    train_data = TensorDataset(data, labels)
    data_loader = DataLoader(train_data, batch_size=batch_size)
    print("- {0} loader done.".format(data_type))
    return feat_size, data_loader


def train(rp: str):
    """
    This method trains the neural network defined in [[NeuralNet]] class.
    1. Get the `train_loader` dataset to be used in PyTorch framework.
    2. Create the NN instance.
    3. Set the `optimizer` to be used.
    4. Set the loss `criterion` to be used.
    5. Loop over the data for the given epochs.

    :argument
        :param rp: Absolute path of the root directory of the project.

    :return:
        None
    """
    start_train = datetime.datetime.now().timestamp()
    print("Training started - Neural Network")
    feat_size, train_loader = get_data_loader(rp, "training")
    net_model = NeuralNet(in_layer=feat_size, out_layer=num_of_classes)
    optimizer = torch.optim.LBFGS(net_model.parameters(), lr=learning_rate, max_iter=5)
    criterion = nn.MultiLabelMarginLoss()
    print("- Optimizer and loss criteria is set.")
    print("- Looping over the data to train the neural network. It will take some time, have patience.")
    for epoch in range(epochs):
        for i, (data, labels) in enumerate(train_loader):
            data_on_dev = data.to(device)
            labels_on_dev = labels.to(device)
            outputs = net_model(data_on_dev)
            loss = criterion(outputs, labels_on_dev)
            optimizer.zero_grad()
            loss.backward()
    torch.save(net_model.state_dict(), read_key("coarse_model", rp + "/{0}".format(nn_model_str)))
    end_train = datetime.datetime.now().timestamp()
    total_train = datetime.datetime.utcfromtimestamp(end_train - start_train)
    print("- Training done : {3} model in {0}h {1}m {2}s"
          .format(total_train.hour, total_train.minute, total_train.second, nn_model_str))


def test(rp: str):
    """
    This method test the pre-trained network.

    :argument
        :param rp: Absolute path of the root directory of the project.

    :return:
        None
    """
    feat_size, test_loader = get_data_loader(rp, "test")
    net_model = NeuralNet(in_layer=feat_size, out_layer=num_of_classes)
    net_model.load_state_dict(torch.load(read_key("coarse_model", rp + "/{0}".format(nn_model_str))))
