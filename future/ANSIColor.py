from dataclasses import dataclass

@dataclass
class ANSIColor:
    """
    Creates an ANSI style terminal color using provided hex color or rgb values
    """
    def __init__(self, text_color:str|tuple=None, text_bold:bool=False):
        """creates a pen styling tool using ansi terminal colors, text_color and background_color must be in rgb or hex format, text_bold is off by default"""
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
        """returns text color attribute as tuple in format of (r, g, b)"""
        return self.text_color_rgb
    
    def alert(self, alerter:str, alert_type:str, bold_alert:bool=False) -> str:
        """issues ANSI color alert on behalf of alerter using specified preset"""
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
        """wraps the provided text in the style of the pen"""
        return f"""{self._ansi_start}{text}{self._ansi_stop}"""
    def wrap_error(self, text:str) -> str:
        """wraps the provided text in the style of the pen, prepending a newline character to print at beginning of stdout"""
        return f"""\r{self._ansi_start}{text}{self._ansi_stop}"""
    def alert_error(self, text:str) -> str:
        return f"""\r\033[1m\033[38;2;247;141;160m{text}\033[0m\033[39m\033[49m"""