
from pathlib import Path

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

def parse_dynamic(data:dict, key:str) -> list:
    vals = []
    count = 1
    while data.get((who := f"{key}_{count}"), None) is not None:
        vals.append(who)
        count += 1
    if len(vals) == 0:
        vals.append([])
    return vals
