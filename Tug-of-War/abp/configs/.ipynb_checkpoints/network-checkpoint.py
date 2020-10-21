from abp.configs import BaseConfig


# TODO
# Maybe use keras instead of reinventing the wheel?
# Input and Output shape should be more customizable
# CNN and other types of network config


default_network_config = {
    "input_shape": [10],  # This is for the shape of the input layer
    "layers": [{"type": "FC", "neurons": 100}] * 2,  # Two fully-connected layers of 100 units each
    "output_shape": [5],  # the number of ouput layers
    "restore_network": True,  # restore from the saved model.
    "network_path": None,  # path used to load/store the network.
    "summaries_path": None,  # path to store the tensorflow summaries for the network
    "summaries_step": 100,  # generates summaries for the network after every n steps
    "networks": [],  # cointaining details of multiple network
    "shared_layers": [],  # the layers shared among the networks TODO
    "aggregator": "average",  # the type of the aggregator TODO
    "save_network": True,
    "save_steps": 100,
    "learning_rate": 0.001,
    "version": "v0"
}


class NetworkConfig(BaseConfig):
    """ Use this object to configure the network for the adpative """

    def __init__(self, config=None):
        super(NetworkConfig, self).__init__(config=config, default_config=default_network_config)

    input_shape = property(BaseConfig.get_property("input_shape"),
                           BaseConfig.set_property("input_shape"))

    layers = property(BaseConfig.get_property("layers"),
                      BaseConfig.set_property("layers"))

    output_shape = property(BaseConfig.get_property("output_shape"),
                            BaseConfig.set_property("output_shape"))

    restore_network = property(BaseConfig.get_property("restore_network"),
                               BaseConfig.set_property("restore_network"))

    network_path = property(BaseConfig.get_property("network_path"),
                            BaseConfig.set_property("network_path"))

    summaries_path = property(BaseConfig.get_property("summaries_path"),
                              BaseConfig.set_property("summaries_path"))

    summaries_step = property(BaseConfig.get_property("summaries_step"),
                              BaseConfig.set_property("summaries_step"))

    networks = property(BaseConfig.get_property("networks"),
                        BaseConfig.set_property("networks"))

    shared_layers = property(BaseConfig.get_property("shared_layers"),
                             BaseConfig.set_property("shared_layers"))

    aggregator = property(BaseConfig.get_property("aggregator"),
                          BaseConfig.set_property("aggregator"))

    save_network = property(BaseConfig.get_property("save_network"),
                            BaseConfig.set_property("save_network"))

    save_steps = property(BaseConfig.get_property("save_steps"),
                          BaseConfig.set_property("save_steps"))

    learning_rate = property(BaseConfig.get_property("learning_rate"),
                             BaseConfig.set_property("learning_rate"))
    
    version = property(BaseConfig.get_property("version"),
                             BaseConfig.set_property("version"))
