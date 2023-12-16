from dataclasses import dataclass

@dataclass
class ANSIColor:
    """
    Creates an ANSI style terminal color using provided hex color or rgb values.

    Attributes:
        - `text_color` (str or tuple): Hex color code or RGB tuple.
        - `text_bold` (bool): Whether text should be bold.

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
        self.text_bold = "\033[1m" if text_bold else ""
        if type(text_color) == str: # assume hex
            fg_r, fg_g, fg_b = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            self.text_color_str = text_color
            self.text_color_hex = text_color
            self.text_color_rgb = (fg_r, fg_g, fg_b)
        if type(text_color) == tuple: # assume rgb
            fg_r, fg_g, fg_b = text_color
            self.text_color_str = str(text_color)
            self.text_color_hex = f"#{fg_r:02x}{fg_g:02x}{fg_b:02x}"
            self.text_color_rgb = (fg_r, fg_g, fg_b)
        self._ansi_start = f"""{self.text_bold}\033[38;2;{fg_r};{fg_g};{fg_b}m"""
        self._ansi_stop = "\033[0m\033[39m\033[49m"

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
    
    def alert(self, alerter:str, alert_type:str, bold_alert:bool=False) -> str:
        """
        Issues an ANSI color alert on behalf of alerter using a specified preset.

        Parameters:
            - `alerter` (str): The entity issuing the alert.
            - `alert_type` (str): Type of alert ('S' for success, 'W' for warning, 'E' for error).
            - `bold_alert` (bool): Whether the alert should be bold (default: False).

        Returns:
            - `str`: ANSI color alert string.

        Example:
        ```python
        import ANSIColor

        # Create the color
        color = ANSIColor("#ff0000")

        # Print an alert message
        print(color.alert("User", "S", bold_alert=True))
        ```        
        """
        match alert_type:
            case 'S': # success
                return f"""{self.text_bold}\033[38;2;108;211;118m{alerter} Success:\033[0m\033[39m\033[49m""" # changed to 108;211;118
            case 'W': # warn
                return f"""{self.text_bold}\033[38;2;246;221;109m{alerter} Warning:\033[0m\033[39m\033[49m"""
            case 'E': # error
                return f"""{self.text_bold}\033[38;2;247;141;160m{alerter} Error:\033[0m\033[39m\033[49m"""
            case other:
                return None

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
    
