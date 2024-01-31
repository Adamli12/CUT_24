# -*- coding: utf-8 -*-


"""
CUT.utils.utils
################################
"""

import importlib

from CUT.utils.enum_type import ModelType


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        'cross_domain_recommender'
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['CUT.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('CUT.trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type == ModelType.CROSSDOMAIN:
            return getattr(importlib.import_module('CUT.trainer'), 'CrossDomainTrainer')
        else:
            return getattr(importlib.import_module('recbole.trainer'), 'Trainer')
