import yaml


class BaseConfig(object):
    """ Extend this object if you want to create config """

    def load_from_yaml(cls, file_path):
        with open(file_path, 'r') as stream:
            config = yaml.load(stream)
            rslt = cls(config=config)
            return rslt

    def __init__(self, config=None, default_config=None):
        super(BaseConfig, self).__init__()
        self._config = config
        self._default_config = default_config

    def _get_property(self, property_name):
        if self._config is None or property_name not in self._config.keys():
            if self._default_config is None or property_name not in self._default_config.keys():
                return None  # just return None if not found
            else:
                return self._default_config[property_name]

        return self._config[property_name]

    def _set_property(self, property_name, value):
        if self._config is None:  # we don't want KeyError
            self._config = {}

        self._config[property_name] = value
        return self._config[property_name]

    def set_property(cls, property_name):
        return (lambda self, value: self._set_property(property_name, value))

    def get_property(clas, property_name):
        return (lambda self: self._get_property(property_name))

    set_property = classmethod(set_property)
    get_property = classmethod(get_property)

    load_from_yaml = classmethod(load_from_yaml)
