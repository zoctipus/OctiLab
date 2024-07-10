from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs.manager_based_rl_env_cfg import ManagerBasedEnvCfg


def create_hebi_env(base_env_cfg: ManagerBasedEnvCfg, rd_action_class):
    """
    Dynamically creates a Hebi environment class based on the base environment and action configuration.

    :param base_env: The base environment class (e.g., IdealPDHebi_JointPos_GoalTracking_Env)
    :param name_suffix: Suffix for the dynamically generated class name to distinguish it.
    :return: A new dynamically created class.
    """
    @configclass
    class DynamicHebiEnv(base_env_cfg):
        def __post_init__(self):
            super().__post_init__()
            self.actions = rd_action_class()

        def __reduce__(self):
            state = self.__dict__.copy()
            return (_NestedClassGetter(),
                    (base_env_cfg, self.__class__.__name__, ),
                    state,
                    )
    new_action_class_name = rd_action_class.__name__.replace("RobotActionsCfg_Hebi", "")
    DynamicHebiEnv.__name__ = f"{base_env_cfg.__name__}_{new_action_class_name}"
    return DynamicHebiEnv


class _NestedClassGetter(object):
    """
    When called with the containing class as the first argument,
    and the name of the nested class as the second argument,
    returns an instance of the nested class.
    """
    def __call__(self, containing_class, class_name):
        nested_class = getattr(containing_class, class_name)

        # make an instance of a simple object (this one will do), for which we can change the
        # __class__ later on.
        nested_instance = _NestedClassGetter()

        # set the class of the instance, the __init__ will never be called on the class
        # but the original state will be set later on by pickle.
        nested_instance.__class__ = nested_class
        return nested_instance
