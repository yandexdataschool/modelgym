from modelgym.models import Model

class ModelSpace(object):
    def __init__(self, model_class, space=None, name=None, space_update=True):
        """
        Args:
            model_class(type): class of model
            space (dict string -> hyperopt distribution or None):
                space of model parameters. If None than default is used
            name (string or None): name of ModelSpace. If None, model
                class name is used
            space_update (bool): whether space param changes default
                model space or replaces it completely
        """
        if not issubclass(model_class, Model):
            raise ValueError("model_class should be subclass of Model")
        self.model_class = model_class
        self.space = model_class.get_default_parameter_space()
        if space is not None:
            if space_update:
                self.space.update(space)
            else:
                self.space = space
        self.name = name
        if self.name is None:
            self.name = model_class.__name__


def process_model_spaces(model_spaces):
    """Process model spaces list (or one model space),

    changing converting Model's to ModelSpaces
    and checking name uniqueness

    Args:
        model_spaces ([list of] Model classes or
            ModelSpaces): list to process
    Returns:
        dict name (str) -> ModelSpace
    """
    if not isinstance(model_spaces, list):
        model_spaces = [model_spaces]

    result = {}
    for model_space in model_spaces:
        if isinstance(model_space, type):
            if not issubclass(model_space, Model):
                raise ValueError("model_spaces should be either Models " +
                                 "classes or ModelSpaces")
            model_space = ModelSpace(model_space)
        elif not isinstance(model_space, ModelSpace):
            raise ValueError("model_spaces should be either Models " +
                             "classes or ModelSpaces")
        if model_space.name in result:
            raise ValueError("Dublicate name: %s" % str(model_space.name))
        result[model_space.name] = model_space
    return result
