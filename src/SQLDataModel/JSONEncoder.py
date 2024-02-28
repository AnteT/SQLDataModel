import json, datetime

class DataTypesEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that extends the functionality of json.JSONEncoder
    to handle additional data types.

    Serialization:
        - `datetime.date`: Serialized as a string in the format 'YYYY-MM-DD'.
        - `datetime.datetime`: Serialized as a string in the format 'YYYY-MM-DD HH:MM:SS'.
        - `bytes`: Decoded to a UTF-8 encoded string.

    Note:
        - The date and datetime types can be recovered using SQLDataModels `infer_dtypes()` method.
        - The bytes information is not decoded back into bytes.
    """    
    def default(self, obj):
        """
        Override the default method to provide custom serialization for specific data types.

        Parameters:
            - `obj`: The Python object to be serialized.

        Returns:
            - The JSON-serializable representation of the object.
        """        
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, bytes):
            return obj.decode(encoding='utf-8')