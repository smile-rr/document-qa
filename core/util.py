def print_structure(obj):
    """
    Utility function to print the open (non-private) and non-function attributes of any Python object.
    """
    print("Type of object:", type(obj))
    
    attributes = dir(obj)
    open_non_function_attributes = {}
    
    for attr in attributes:
        if not attr.startswith('_'):
            value = getattr(obj, attr)
            if not callable(value):
                open_non_function_attributes[attr] = value
    
    print("Open non-function attributes of object:")
    for attr, value in open_non_function_attributes.items():
        print(f"{attr}: {value}")

# Example usage
if __name__ == "__main__":
    class Example:
        def __init__(self):
            self.attribute = "value"
            self._private_attribute = "private"
        
        def method(self):
            pass
    
    example_instance = Example()
    print_structure(example_instance)