
import unicodedata
from pathlib import Path

from PIL import ImageDraw

try:
    # khatt joins Arabic letters and lays them out right to left, which Pillow only
    # does when it was built with Raqm (not on Windows); other text goes to Pillow.
    from khatt.pillow import text as draw_text
    from khatt.pillow import textbbox as text_bbox
except ImportError:  # without khatt, Pillow draws everything as before
    draw_text = ImageDraw.ImageDraw.text
    text_bbox = ImageDraw.ImageDraw.textbbox

### GLOBALS

ROOT = Path(__file__).absolute().parent.parent
ROOT_COMFY = ROOT.parent.parent
ROOT_FONTS = ROOT / "fonts"

### SUPPORT CLASSES

class AlwaysEqualProxy(str):
    def __eq__(self, other):
        return True
    def __ne__(self, other):
        return False

### SUPPORT FUNCTIONS

def has_arabic(text: str) -> bool:
    """Whether text has Arabic letters, which draw_text lays out with khatt."""
    return any(unicodedata.bidirectional(char) == "AL" for char in text)

def parse_dynamic(data:dict, key:str) -> list:
    vals = []
    count = 1
    while data.get((who := f"{key}_{count}"), None) is not None:
        vals.append(who)
        count += 1
    if len(vals) == 0:
        vals.append([])
    return vals
