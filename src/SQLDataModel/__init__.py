from SQLDataModel.SQLDataModel import SQLDataModel
from SQLDataModel.converters import register_adapters_and_converters

register_adapters_and_converters()
del(register_adapters_and_converters)