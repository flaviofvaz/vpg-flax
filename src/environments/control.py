from dm_control import suite
import numpy as np


def load_environment(env_name: str, task_name: str) -> suite.control.Environment:
    return suite.load(domain_name=env_name, task_name=task_name)
