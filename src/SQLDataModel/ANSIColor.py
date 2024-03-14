from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ANSIColor:
    """
    Creates an ANSI style terminal color using provided hex color or rgb values.

    Attributes:
        - `text_color` (str or tuple): Hex color code or RGB tuple.
        - `text_bold` (bool): Whether text should be bold.

    Raises:
        - `ValueError`: If provided string is not a valid hex color code or if provided rgb tuple is invalid.
        - `TypeError`: If provided `text_color` or `text_bold` parameters are of invalid types.

    Methods:
        - `__init__`: Initializes the ANSIColor object with the specified text color and bold setting.
        - `__repr__`: Returns a string representation of the ANSIColor object with ANSI escape codes.
        - `to_rgb`: Returns the text color attribute as a tuple in the format (r, g, b).
        - `alert`: Issues an ANSI color alert using a specified preset.
        - `wrap`: Wraps the provided text in the style of the pen.
        - `wrap_error`: Wraps the provided text in the style of the pen, prepending a newline character.
        - `alert_error`: Creates an ANSI color alert for error messages.

    Example:
    ```python
    import ANSIColor

    # Create a pen by specifying a color in hex or rgb:
    green_bold = ANSIColor("#00ff00", text_bold=True)

    # Create a string to use as a sample:
    regular_str = "Hello, World!"

    # Color the string using the `wrap()` method:
    green_str = green_bold.wrap(regular_str)

    # Print the string in the terminal to see the color applied:
    print(f"original string: {regular_str}, green string: {green_str}")

    # Get rgb values from existing color
    print(green_bold.to_rgb())  # Output: (0, 255, 0)
    ```
    """
    def __init__(self, text_color:str|tuple=None, text_bold:bool=False):
        """
        Initializes the ANSIColor object with the specified text color and bold setting.

        Parameters:
            - `text_color` (str or tuple): Hex color code or RGB tuple (default: None, teal color used if not specified)
            - `text_bold` (bool): Whether text should be bold (default: False)

        Example:
        ```python
        import ANSIColor

        # Initialize from hex value with normal weight
        color = ANSIColor("#00ff00")

        # Initialize from rgb value with bold weight
        color = ANSIColor((0,255,0), text_bold=True)
        ```
        """
        text_color = (95, 226, 197) if text_color is None else text_color # default teal color
        if not isinstance(text_color, (tuple,str)):
            raise TypeError(
                ANSIColor.ErrorFormat(f"TypeError: invalid `text_color` type '{type(text_color).__name__}' received, expected value of type 'tuple' or 'str'")
            )
        self.text_bold = "\033[1m" if text_bold else ""
        if type(text_color) == str: # assume hex
            fg_r, fg_g, fg_b = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            self.text_color_str = text_color
            self.text_color_hex = text_color
            self.text_color_rgb = (fg_r, fg_g, fg_b)
            for color_value in self.text_color_rgb:
                if color_value < 0 or color_value > 255:
                    raise ValueError(
                        ANSIColor.ErrorFormat(f"ValueError: invalid value '{color_value}' in rgb color {self.text_color_rgb}, all values must satisfy '0 <= value <= 255'")
                    )
        if type(text_color) == tuple: # assume rgb
            fg_r, fg_g, fg_b = text_color
            self.text_color_str = str(text_color)
            self.text_color_hex = f"#{fg_r:02x}{fg_g:02x}{fg_b:02x}"
            self.text_color_rgb = (fg_r, fg_g, fg_b)
            for color_value in self.text_color_rgb:
                if color_value < 0 or color_value > 255:
                    raise ValueError(
                        ANSIColor.ErrorFormat(f"ValueError: invalid value '{color_value}' in rgb color {self.text_color_rgb}, all values must satisfy '0 <= value <= 255'")
                    )            
        self._ansi_start = f"""{self.text_bold}\033[38;2;{fg_r};{fg_g};{fg_b}m"""
        self._ansi_stop = "\033[0m\033[39m\033[49m"

    @staticmethod
    def ErrorFormat(error:str) -> str:
        """
        Formats an error message with ANSI color coding.

        Parameters:
            - `error`: The error message to be formatted.

        Returns:
            - A string with ANSI color coding, highlighting the error type in bold red.

        Example:
        ```python
        formatted_error = ErrorFormat("ValueError: Invalid value provided.")
        print(formatted_error)
        ```
        """
        error_type, error_description = error.split(':',1)
        return f"""\r\033[1m\033[38;2;247;141;160m{error_type}:\033[0m\033[39m\033[49m{error_description}"""    

    def __repr__(self) -> str:
        return f"""{self._ansi_start}{type(self).__name__}({self.text_color_str}){self._ansi_stop}"""

    def to_rgb(self) -> tuple:
        """
        Returns the text color attribute as a tuple in the format (r, g, b).

        Returns:
            - `tuple`: RGB tuple.

        Example:
        ```python
        import ANSIColor

        # Create the color
        color = ANSIColor("#00ff00")

        # Get the rgb values
        print(color.to_rgb())  # Output: (0, 255, 0)
        ```
        """
        return self.text_color_rgb
    
    def wrap(self, text:str) -> str:
        """
        Wraps the provided text in the style of the pen.

        Parameters:
            - `text` (str): Text to be wrapped.

        Returns:
            - `str`: Wrapped text with ANSI escape codes.

        Example:
        ```python
        import ANSIColor

        # Create the color
        blue_color = ANSIColor("#0000ff")

        # Create a sample string
        message = "This string is currently unstyled"

        # Wrap the string to change its styling whenever its printed
        blue_message = blue_color.wrap(message)

        # Print the styled message
        print(blue_message)

        # Or style string or string object directly in the print statement
        print(blue_color.wrap("I'm going to turn blue!"))
        
        ```
        """
        return f"""{self._ansi_start}{text}{self._ansi_stop}"""
    
    def wrap_error(self, text:str) -> str:
        """
        Wraps the provided text in the style of the pen, prepending a newline character.

        Parameters:
            - `text` (str): Text to be wrapped.

        Returns:
            - `str`: Wrapped text with ANSI escape codes and a prepended newline character.
        """
        return f"""\r{self._ansi_start}{text}{self._ansi_stop}"""
    
    def alert_error(self, text:str) -> str:
        """
        Creates an ANSI color alert for error messages.

        Parameters:
            - `text` (str): Error message.

        Returns:
            - `str`: ANSI color alert string for error messages.
        """        
        return f"""\r\033[1m\033[38;2;247;141;160m{text}\033[0m\033[39m\033[49m"""