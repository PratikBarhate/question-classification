from configobj import ConfigObj


# return value for a given key from conf file
def read_file(file_name):
    config = ConfigObj('../resources/properties.conf')
    value = config[file_name]
    return value


print(read_file("training_data"))