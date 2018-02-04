from configobj import ConfigObj
import os
import pickle


text_file_encoding = "utf8"


def read_key(file_name):
    """
    :argument:
        :param file_name: String - Key to be fetched from properties.conf
    :return:
        value: String - value defined in properties.conf for the given key.
    """
    config = ConfigObj("../../resources/properties.conf")
    value = config[file_name]
    return value


def read_file(file_name):
    """
    :argument:
        :param file_name: A string which represents the raw data file, in properties.conf,
                          used for the process (experiment).
        # Note: File should be using UTF-8 encoding. Change the encoding as needed.
    :exception:
        :except IOError: This may occur because of many reasons. e.g file is missing or corrupt file or wrong file path
    :return:
        boolean_flag: True for successful read operation.
        file: TextIOWrapper for the file corresponding to the `file_name` key in properties.conf
    """
    try:
        file = open(read_key(file_name), "r", encoding=text_file_encoding)
        return True, file
    except IOError as e:
        print("File IO Error :: Cannot open " + read_key(file_name) + "\n" + str(e))
        return False


def write_str_file(str_list, file_name):
    """
    :argument:
        :param str_list: List of string which are to written to the file with line endings
        :param file_name: String which represents the file from properties.conf
    :return:
        boolean flag: True for successful operation.
    """
    file_value = read_key(file_name)
    if not os.path.exists(os.path.dirname(file_value)):
        try:
            os.makedirs(os.path.dirname(file_value))
        except OSError as err:
            print("Error creating director " + file_value + "\n" + str(err))
            return False
    try:
        with open(file_value, "w", encoding=text_file_encoding) as file:
            str_to_write = "\n".join(str_list)
            file.write(str_to_write)
    except IOError as e:
        print("File IO Error :: Cannot write text(string) to " + file_value + "\n" + str(e))
        return False
    return True


def write_obj(obj, file_name):
    """
    :argument:
        :param obj: Object to br written to the disk at the given location
        :param file_name: String which represents the file from properties.conf
    :return:
        boolean flag: True for successful operation.
    """
    file_value = read_key(file_name)
    if not os.path.exists(os.path.dirname(file_value)):
        try:
            os.makedirs(os.path.dirname(file_value))
        except OSError as err:
            print("Error creating director " + file_value + "\n" + str(err))
            return False
    try:
        with open(file_value, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    except IOError as e:
        print("File IO Error :: Cannot write object to " + file_value + "\n" + str(e))
        return False
    return True
