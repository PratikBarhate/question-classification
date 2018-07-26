import os
import pickle

from configobj import ConfigObj

# Every write operation is in overwrite mode
text_file_encoding = "utf8"


def read_key(file_key: str, rp: str):
    """
    :argument:
        :param file_key: String - Key to be fetched from properties.conf
        :param rp: Absolute path of the root directory of the project
    :return:
        value: String - value defined in properties.conf for the given key, appended to the root path of the project.
    """
    if file_key.endswith("vec") or file_key.endswith("model") or file_key.endswith("binarizer"):
        res_p = "/".join(rp.split("/")[:-1])
    else:
        res_p = rp
    config = ConfigObj("{0}/resources/properties.conf".format(res_p))
    value = rp + "/" + config[file_key]
    return value


def read_file(file_key, rp):
    """
    :argument:
        :param file_key: A string which represents the raw data file, in properties.conf,
                          used for the process (experiment).
        :param rp: Absolute path of the root directory of the project
        # Note: File should be using UTF-8 encoding. Change the encoding as needed.
    :exception:
        :except IOError: This may occur because of many reasons. e.g file is missing or corrupt file or wrong file path
    :return:
        boolean_flag: True for successful read operation.
        file: TextIOWrapper for the file corresponding to the `file_name` key in properties.conf
    """
    try:
        file = open(read_key(file_key, rp), "r", encoding=text_file_encoding)
        return True, file
    except IOError as e:
        print("File IO Error :: Cannot open " + read_key(file_key, rp) + "\n" + str(e))
        return False


def write_str_file(str_list, file_key, rp):
    """
    :argument:
        :param str_list: List of string which are to written to the file with line endings
        :param file_key: String which represents the file from properties.conf
        :param rp: Absolute path of the root directory of the project
    :return:
        boolean flag: True for successful operation.
    """
    key_value: str = read_key(file_key, rp)
    if not os.path.exists(os.path.dirname(key_value)):
        try:
            os.makedirs(os.path.dirname(key_value))
        except OSError as err:
            print("Error creating director " + key_value + "\n" + str(err))
            return False
    try:
        with open(key_value, "w", encoding=text_file_encoding) as file:
            str_to_write = "\n".join(str_list)
            file.write(str_to_write)
    except IOError as e:
        print("File IO Error :: Cannot write text(string) to " + key_value + "\n" + str(e))
        return False
    return True


def write_obj(obj, file_key, rp):
    """
    :argument:
        :param obj: Object to br written to the disk at the given location
        :param file_key: String which represents the file from properties.conf
        :param rp: Absolute path of the root directory of the project
    :return:
        boolean flag: True for successful operation.
    """
    key_value = read_key(file_key, rp)
    if not os.path.exists(os.path.dirname(key_value)):
        try:
            os.makedirs(os.path.dirname(key_value))
        except OSError as err:
            print("Error creating director " + key_value + "\n" + str(err))
            return False
    try:
        with open(key_value, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    except IOError as e:
        print("File IO Error :: Cannot write object to " + key_value + "\n" + str(e))
        return False
    return True


def read_obj(file_key, rp):
    """
    :argument:
        :param file_key: A string which represents the raw data file, in properties.conf,
                          used for the process (experiment).
        :param rp: Absolute path of the root directory of the project
        # Note: File should be using UTF-8 encoding. Change the encoding as needed.
    :exception:
        :except IOError: This may occur because of many reasons. e.g file is missing or corrupt file or wrong file path
    :return:
        boolean_flag: True for successful read operation.
        file: TextIOWrapper for the file corresponding to the `file_name` key in properties.conf
    """
    try:
        file = open(read_key(file_key, rp), "rb")
        obj = pickle.load(file)
        return True, obj
    except IOError as e:
        print("File IO Error :: Cannot open " + read_key(file_key, rp) + "\n" + str(e))
        return False
