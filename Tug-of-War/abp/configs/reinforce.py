from abp.configs.base_config import BaseConfig

default_reinforce_config = {
    "epsilon_timesteps": 1000,
    "starting_epsilon": 1.0,
    "final_epsilon": 0.1,
    "discount_factor": 0.95,
    "batch_size": 35,
    "memory_size": 10000,  # Total Memory size for the batch
    "summaries_path": None,  # Path to store reinforcement related summaries
    "replace_frequency": 1000,
    "update_steps": 1,  # Will update the model after n steps
    "update_start": 1,  # Will start update after n steps
    "beta_timesteps": 10000,
    "beta_initial": 0.2,
    "beta_final": 1.0,
    "decay_steps": 250,
    "decay_rate": .96,
    "use_prior_memory" : True,
    "is_random_agent_1" : False,
    "is_random_agent_2" : False,
    "collecting_experience": False,
    "is_use_sepcific_enemy": False,
    "enemy_path": None
}


class ReinforceConfig(BaseConfig):
    """Configurations related to reinfocement learning"""

    def __init__(self, config=None):
        super(ReinforceConfig, self).__init__(config,
                                              default_config=default_reinforce_config)

    # TODO Can use python feature to generate these properties? Manual Labour for now -_-

    epsilon_timesteps = property(BaseConfig.get_property("epsilon_timesteps"),
                                 BaseConfig.set_property("epsilon_timesteps"))

    starting_epsilon = property(BaseConfig.get_property("starting_epsilon"),
                                BaseConfig.set_property("starting_epsilon"))

    final_epsilon = property(BaseConfig.get_property("final_epsilon"),
                             BaseConfig.set_property("final_epsilon"))

    discount_factor = property(BaseConfig.get_property("discount_factor"),
                               BaseConfig.set_property("discount_factor"))

    batch_size = property(BaseConfig.get_property("batch_size"),
                          BaseConfig.set_property("batch_size"))

    memory_size = property(BaseConfig.get_property("memory_size"),
                           BaseConfig.set_property("memory_size"))

    summaries_path = property(BaseConfig.get_property("summaries_path"),
                              BaseConfig.set_property("summaries_path"))

    replace_frequency = property(BaseConfig.get_property("replace_frequency"),
                                 BaseConfig.set_property("replace_frequency"))

    update_steps = property(BaseConfig.get_property("update_steps"),
                            BaseConfig.set_property("update_steps"))

    update_start = property(BaseConfig.get_property("update_start"),
                            BaseConfig.set_property("update_start"))

    beta_timesteps = property(BaseConfig.get_property("beta_timesteps"),
                              BaseConfig.set_property("beta_timesteps"))

    beta_initial = property(BaseConfig.get_property("beta_initial"),
                            BaseConfig.set_property("beta_initial"))

    beta_final = property(BaseConfig.get_property("beta_final"),
                          BaseConfig.set_property("beta_final"))

    decay_steps = property(BaseConfig.get_property("decay_steps"),
                          BaseConfig.set_property("decay_steps"))

    decay_rate = property(BaseConfig.get_property("decay_rate"),
                          BaseConfig.set_property("decay_rate"))
    
    use_prior_memory = property(BaseConfig.get_property("use_prior_memory"),
                          BaseConfig.set_property("use_prior_memory"))
    
    is_random_agent_1 = property(BaseConfig.get_property("is_random_agent_1"),
                          BaseConfig.set_property("is_random_agent_1"))
    
    is_random_agent_2 = property(BaseConfig.get_property("is_random_agent_2"),
                          BaseConfig.set_property("is_random_agent_2"))

    collecting_experience = property(BaseConfig.get_property("collecting_experience"),
                        BaseConfig.set_property("collecting_experience"))
    
    is_use_sepcific_enemy = property(BaseConfig.get_property("is_use_sepcific_enemy"),
                        BaseConfig.set_property("is_use_sepcific_enemy"))

    enemy_path = property(BaseConfig.get_property("enemy_path"),
                        BaseConfig.set_property("enemy_path"))