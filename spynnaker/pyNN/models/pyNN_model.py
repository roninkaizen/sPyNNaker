from spynnaker.pyNN import exceptions
import inspect

_REQUIRED_PROPERTIES = [
    "binary_name",
    "supports_connector"
]

# The expected attributes and types of a PyNN model
_PROPERTIES = {
    "default_parameters": dict,
    "state_variables": list,
    "fixed_parameters": dict,
    "recording_types": list,
    "is_array_parameters": list,
    "population_parameters": list,
    "binary_name": str,
    "max_atoms_per_core": int,
    "supports_connector": bool
}

# The arguments of the create_vertex method
_CREATE_VERTEX_ARGS = [
    "population_parameters", "neuron_cells", "label",
    "constraints"
]


class classproperty(property):
    """ Defines a property of the class
    """

    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class staticproperty(property):
    """ Defines a static property
    """

    def __get__(self, cls, owner):
        return staticmethod(self.fget).__get__(None, owner)()


def pyNN_model(model_class):
    """ Indicates that the class is a PyNN model i.e. something that can be\
        specified as a model class in a PyNN population.  The model is checked\
        for "n_neurons" as a parameter of the constructor, and that it\
        contains the other static requirements of a PyNN model
    """

    for attribute in _REQUIRED_PROPERTIES:
        if not hasattr(model_class, attribute):
            raise exceptions.SpynnakerException(
                "The model {} does not have the required property {}".format(
                    model_class, attribute))

    for attribute, attribute_type in _PROPERTIES.iteritems():
        if hasattr(model_class, attribute):
            attr = getattr(model_class, attribute)
            if not isinstance(attr, attribute_type):
                raise exceptions.SpynnakerException(
                    "The attribute {} of the model {} is not a {}".format(
                        attribute, model_class, attribute_type))

    if not hasattr(model_class, "create_vertex"):
        raise exceptions.SpynnakerException(
            "The model {} is missing a create_vertex static function".format(
                model_class))
    create_vertex_method = model_class.create_vertex
    if not inspect.ismethod(create_vertex_method):
        raise exceptions.SpynnakerException(
            "create_vertex is not a method in the model {}".format(
                model_class))
    argspec = inspect.getargspec(create_vertex_method)
    for arg in _CREATE_VERTEX_ARGS:
        if arg not in argspec.args:
            raise exceptions.SpynnakerException(
                "The create_vertex method of the model {} has no parameter {}"
                .format(model_class, arg))

    # Mark the model as a PyNN model
    model_class.__is_pyNN_model__ = True

    return model_class
