from __future__ import annotations
import random

class ANSIColor:
    """
    Creates an ANSI style terminal color using provided hex color or rgb values.

    Attributes:
        ``text_color`` (str or tuple): Hex color code or RGB tuple.
        ``text_bold`` (bool): Whether text should be bold.

    Raises:
        ``ValueError``: If provided string is not a valid hex color code or if provided rgb tuple is invalid.
        ``TypeError``: If provided text_color or text_bold parameters are of invalid types.

    Example::

        import ANSIColor

        # Create a pen by specifying a color in hex or rgb:
        green_bold = ANSIColor("#00ff00", text_bold=True)

        # Create a string to use as a sample:
        regular_str = "Hello, World!"

        # Color the string using the wrap method:
        green_str = green_bold.wrap(regular_str)

        # Print the string in the terminal to see the color applied:
        print(f"original string: {regular_str}, green string: {green_str}")

        # Get rgb values from existing color
        print(green_bold.to_rgb())  # Output: (0, 255, 0)

    Changelog:
        - Version 0.10.2 (2024-06-30):
            - Added random color selection when initialized without a ``text_color`` argument.
            - Added dictionary of color values at :py:attr:`ANSIColor.Colors` to use as selection pool.
            - Modified :meth:`ANSIColor.__repr__()` to always return hex value as a string for consistency regardless of original input format.
    """

    Colors = {
        'gray-100': '#FAFBFD'
        ,'purple-300': '#B86BA2'
        ,'purple-350': '#B09CF1'
        ,'blue-200': '#A3D7E7'
        ,'blue-300': '#51BDDF'
        ,'purple100': '#DFD8F3'
        ,'purple300': '#B39CF1'
        ,'red-105': '#E6CED2'
        ,'red-115': '#DDBAC0'
        ,'red-150': '#F78DA0'
        ,'orange-100': '#FBC375'
        ,'orange-125': '#EAC69E'
        ,'orange-150': '#EFAC65'
        ,'orange-200': '#F9B148'
        ,'orange-400': '#DF8607'
        ,'yellow-125': '#F5E7A9'
        ,'yellow-150': '#F6DD6D'
        ,'yellow-200': '#9ACA42'
        ,'yellow-400': '#F5CC0E'
        ,'green-100': '#ABCD92'
        ,'green-200': '#6CD376'
        ,'teal-100': '#8BE3E9'
        ,'teal-150': '#8BBDB8'
        ,'brown-150': '#C3B7B4'
        ,'sand-150': '#D6D0B9'
        ,'slate-150': '#B6BECC'
        ,'amber-100': '#F3C47F'
        ,'amber-200': '#fac000'
        ,'bronze-150': '#D0B196'
        ,'coral-150': '#DBA4A4'
        ,'olive-150': '#AEBC83'
        ,'cyan-150': '#A6D7E8'
        ,'cyan-300': '#78BED7'
        ,'ocean-150': '#9FBADA'
        ,'plum-150': '#C8A4C6'
        ,'pink-100': '#F188C0'
        ,'pink-500': '#E94B9F'
        ,'aqua-100': '#24C9FF'
        ,'peach-100': '#FFB8BA'
        ,'peach-500': '#E87276'
        ,'peach-900': '#B94448'
        ,'violet-100': '#A6ACD8'
        ,'violet-500': '#8655ED'
        ,'emerald-100': '#A3E3A7'
        ,'cyan-100': '#71D9E7'
        ,'amber-500': '#D18411'
        ,'indigo-100': '#97A8DA'
        ,'olive-100': '#C0B97C'
        ,'purple-100': '#D7B1D0'
        ,'rose-100': '#F2A9B9'
        ,'rose-500': '#E8728A'
        ,'plum-100': '#D9A4E2'
        ,'plum-500': '#BA66C8'
        ,'plum-900': '#9B27AF'
        ,'sky-100': '#98D5DD'
        ,'orange-300': '#F89F1F'
        ,'yellow-100': '#B5D874'
        ,'yellow-300': '#CBBE06'
        ,'teal-200': '#56D6DF'
        ,'teal-300': '#20AF8F'
        ,'lightblue-100': '#A7D9FF'
        ,'lightblue-200': '#7AC6FF'
        ,'lightblue-300': '#47B0FF'
        ,'aqua-200': '#00B1EB'
        ,'violet-200': '#B3A2CD'
        ,'purple-200': '#C389B7'
        ,'magenta-100': '#F6A8CC'
        ,'magenta-200': '#EE63A4'
        ,'magenta-300': '#D6297A'
        ,'magenta-400': '#B61B64'
        ,'blue-100': '#B3DEEB'
        ,'red-100': '#D595B1'
        ,'red-200': '#D0719B'
        ,'red-300': '#B94D7D'
        ,'blue-50': '#D7EFF8'
        ,'blue-75': '#C9E7F1'
        ,'blue-225': '#90D5EA'
        ,'violet-150': '#ADA6E8'
    }
    """``dict[str, str]``: A dictionary of preselected colors with format ``{'label': 'hexcode'}`` to use as selection pool for :meth:`ANSIColor.rand_color()` for SQLDataModel."""
    
    def __init__(self, text_color:str|tuple=None, text_bold:bool=False) -> None:
        """
        Initializes the ``ANSIColor`` object with the specified text color and bold setting, referred to as the 'pen' throughout documentation.

        Parameters:
            ``text_color`` (str or tuple): Hex color code or RGB tuple. If not provided, a random color will be selected.
            ``text_bold`` (bool): Whether text should be bold (default: False)

        Example::

            import ANSIColor

            # Initialize from hex value with normal weight
            color = ANSIColor("#00ff00")

            # Initialize from rgb value with bold weight
            color = ANSIColor((0,255,0), text_bold=True)

            # Surprise me! Initialize pen from random color
            color = ANSIColor()

        Changelog:
            - Version 0.10.2 (2024-06-30):
                - Modified to randomly select a color from :py:attr:`ANSIColor.Colors` when ``text_color = None`` for demonstration purposes.
        
        Note:
            - The string returned by :meth:`ANSIColor.__repr__()` will always return the hex value of the pen regardless of the ``text_color`` format.
            - See :py:attr:`ANSIColor.text_color_str` to view originally provided format of ``text_color``.
            - See :py:attr:`ANSIColor.text_color_rgb` to view the RGB tuple equivalent of ``text_color``.
            - See :py:attr:`ANSIColor.text_color_hex` to view the hex value equivalent of ``text_color``.
        """
        text_color = ANSIColor.Colors[random.choice(list(ANSIColor.Colors.keys()))] if text_color is None else text_color # Use random color if None
        if not isinstance(text_color, (tuple,list,str)):
            raise TypeError(
                ANSIColor.ErrorFormat(f"TypeError: invalid `text_color` type '{type(text_color).__name__}' received, expected value of type 'tuple' or 'str'")
            )
        self.text_bold = "\033[1m" if text_bold else ""
        if isinstance(text_color, str): # assume hex
            try:
                fg_r, fg_g, fg_b = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            except:
                raise ValueError(
                    ANSIColor.ErrorFormat(f"ValueError: invalid value '{text_color}', string argument for `text_color` must be a valid hexadecimal value between `#000000` and `#ffffff`")
                ) from None
            self.text_color_str = text_color
            """``str``: The input color used to create the pen in the originally provided format."""
            self.text_color_hex = text_color.upper()
            """``str``: The hex value of color uppercased and prepended with '#' to reflect hexadecimal format ranging from ``'#000000'`` to ``'#FFFFFF'``."""
            self.text_color_rgb = (fg_r, fg_g, fg_b)
            """``tuple[int, int, int]``: The RGB value of the color as a tuple of integers reflecting the (red, green, blue) values satisfying ``0 <= value <= 255``."""
            for color_value in self.text_color_rgb:
                if color_value < 0 or color_value > 255:
                    raise ValueError(
                        ANSIColor.ErrorFormat(f"ValueError: invalid value '{color_value}' in rgb color '{self.text_color_rgb}', all values must be in range `0 <= value <= 255`")
                    )
        if isinstance(text_color, (tuple, list)): # assume rgb
            try:
                fg_r, fg_g, fg_b = text_color
            except:
                raise ValueError(
                    ANSIColor.ErrorFormat(f"ValueError: invalid value '{text_color}', tuple argument for `text_color` must be a valid rgb tuple `(r, g, b)` with values between `0` and `255`")
                ) from None  
            self.text_color_str = str(text_color)
            """``str``: The input color used to create the pen in the originally provided format."""
            self.text_color_hex = f"#{fg_r:02X}{fg_g:02X}{fg_b:02X}"
            """``str``: The hex value of color uppercased and prepended with '#' to reflect hexadecimal format ranging from ``'#000000'`` to ``'#FFFFFF'``."""
            self.text_color_rgb = (fg_r, fg_g, fg_b)
            """``tuple[int, int, int]``: The RGB value of the color as a tuple of integers reflecting the (red, green, blue) values satisfying ``0 <= value <= 255``."""
            for color_value in self.text_color_rgb:
                if color_value < 0 or color_value > 255:
                    raise ValueError(
                        ANSIColor.ErrorFormat(f"ValueError: invalid value '{color_value}' in rgb color '{self.text_color_rgb}', all values must be in range `0 <= value <= 255`")
                    )            
        self._ansi_start = f"""{self.text_bold}\033[38;2;{fg_r};{fg_g};{fg_b}m"""
        self._ansi_stop = "\033[0m\033[39m\033[49m"

    @staticmethod
    def ErrorFormat(error:str) -> str:
        """
        Formats an error message with ANSI color coding.

        Parameters:
            ``error`` (str): The error message to be formatted.

        Returns:
            ``str``: A string with ANSI color coding, highlighting the error type in bold red.
        
        Example::
            
            import ANSIColor

            # Error message to format
            formatted_error = ANSIColor.ErrorFormat("ValueError: Invalid value provided.")
            
            # Display alongside error or exception when raised
            print(formatted_error)

        """
        error_type, error_description = error.split(':',1)
        return f"""\r\033[1m\033[38;2;247;141;160m{error_type}:\033[0m\033[39m\033[49m{error_description}"""    

    @classmethod
    def rand_color(cls) -> ANSIColor:
        """
        Create a new ANSIColor pen by randomly selecting one from a preexisting pool of options.
        
        Returns:
            ``ANSIColor``: A new ANSIColor instance created using a randomly selected color.

        Example::

            import ANSIColor

            # Surprise me!
            rand_color = ANSIColor.rand_color()

            # See what we got
            print(rand_color)

        We got a nice orance color with this hex value:

        ```text
            ANSIColor('#F89F1F')
        ```

        Note:
            - See :py:attr:`ANSIColor.Colors` for dictionary of values being used as random color selection pool.

        .. versionadded:: 0.10.2
            Added to allow a random color to be selected for :mod:`SQLDataModel.SQLDataModel.set_display_color()`
        """
        rand_color = ANSIColor.Colors[random.choice(list(ANSIColor.Colors.keys()))]
        return cls(rand_color)

    def __repr__(self) -> str:
        """
        The string representation used for instances of ``ANSIColor`` displayed with the pen set at :py:attr:`ANSIColor.text_color_str` formatted to allow object recreation.

        Returns:
            ``str``: The string representation as ``ANSIColor('hexvalue')`` colored with the ANSI terminal color

        Example::

            import ANSIColor

            # Create the pen from a hex value
            color = ANSIColor('#EFAC65') 
            
            # View representation
            print(color)

        This will output:

        ```text
            ANSIColor('#EFAC65')
        ```

        Creating a pen using the equivalent RGB tuple results in the same output:

        ```python
            # From the RGB equivalent values
            color = ANSIColor((239, 172, 101))

            # View representation
            print(color)
        ```
        
        This will also output:

        ```text
            ANSIColor('#EFAC65')
        ```

        Note:
            - The representation will always be formatted using the hex value for consistency and recreation.
            - Use :meth:`ANSIColor.to_rgb()` to view the RGB values for an existing pen.
        """
        return f"""{self._ansi_start}{type(self).__name__}('{self.text_color_hex}'){self._ansi_stop}"""

    def to_rgb(self) -> tuple:
        """
        Returns the text color attribute as a tuple in the format (r, g, b).

        Returns:
            ``tuple``: RGB tuple.

        Example::
        
            import ANSIColor

            # Create the color
            color = ANSIColor("#00ff00")

            # Get the rgb values
            print(color.to_rgb())  # Output: (0, 255, 0)
        
        """
        return self.text_color_rgb
    
    def wrap(self, text:str) -> str:
        """
        Wraps the provided text in the style of the pen.

        Parameters:
            ``text`` (str): Text to be wrapped.

        Returns:
            ``str``: Wrapped text with ANSI escape codes.

        Example::
        
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
        
        """
        return f"""{self._ansi_start}{text}{self._ansi_stop}"""