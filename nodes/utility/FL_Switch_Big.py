import torch
from ..sup import AlwaysEqualProxy

def MakeSmartType(t):
    if isinstance(t, str):
        return SmartType(t)
    return t

class SmartType(str):
    def __ne__(self, other):
        if self == "*" or other == "*":
            return False
        selfset = set(self.split(','))
        otherset = set(other.split(','))
        return not selfset.issubset(otherset)

def VariantSupport():
    def decorator(cls):
        if hasattr(cls, "INPUT_TYPES"):
            old_input_types = getattr(cls, "INPUT_TYPES")
            def new_input_types(*args, **kwargs):
                types = old_input_types(*args, **kwargs)
                for category in ["required", "optional"]:
                    if category not in types:
                        continue
                    for key, value in types[category].items():
                        if isinstance(value, tuple):
                            types[category][key] = (MakeSmartType(value[0]),) + value[1:]
                return types
            setattr(cls, "INPUT_TYPES", new_input_types)
        if hasattr(cls, "RETURN_TYPES"):
            old_return_types = cls.RETURN_TYPES
            setattr(cls, "RETURN_TYPES", tuple(MakeSmartType(x) for x in old_return_types))
        if hasattr(cls, "VALIDATE_INPUTS"):
            # Reflection is used to determine what the function signature is, so we can't just change the function signature
            raise NotImplementedError("VariantSupport does not support VALIDATE_INPUTS yet")
        else:
            def validate_inputs(input_types):
                inputs = cls.INPUT_TYPES()
                for key, value in input_types.items():
                    if isinstance(value, SmartType):
                        continue
                    if "required" in inputs and key in inputs["required"]:
                        expected_type = inputs["required"][key][0]
                    elif "optional" in inputs and key in inputs["optional"]:
                        expected_type = inputs["optional"][key][0]
                    else:
                        expected_type = None
                    if expected_type is not None and MakeSmartType(value) != expected_type:
                        return f"Invalid type of {key}: {value} (expected {expected_type})"
                return True
            setattr(cls, "VALIDATE_INPUTS", validate_inputs)
        return cls
    return decorator

@VariantSupport()
class FL_Switch_Big:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch_condition": ("STRING", {"default": "", "multiline": False}),
                "case_1": ("STRING", {"default": "", "multiline": False}),
                "case_2": ("STRING", {"default": "", "multiline": False}),
                "case_3": ("STRING", {"default": "", "multiline": False}),
                "case_4": ("STRING", {"default": "", "multiline": False}),
                "case_5": ("STRING", {"default": "", "multiline": False}),
                "input_default": ("*", {"lazy": True}),
            },
            "optional": {
                "input_1": ("*", {"lazy": True}),
                "input_2": ("*", {"lazy": True}),
                "input_3": ("*", {"lazy": True}),
                "input_4": ("*", {"lazy": True}),
                "input_5": ("*", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch_case"
    CATEGORY = "🏵️Fill Nodes/Utility"
    
    DESCRIPTION = """
    FL_Switch_Big allows you to choose between multiple processing paths based on a switch condition.
    The node only evaluates the path that is selected by the switch condition.
    
    - The switch_condition is compared against each case value
    - If a match is found, the corresponding input is evaluated and returned
    - If no match is found, input_default is evaluated and returned
    - Inputs that don't match the condition are not evaluated
    
    This node is useful for creating conditional workflows with multiple branches.
    """

    def check_lazy_status(self, switch_condition, case_1, case_2, case_3, case_4, case_5, 
                         input_default=None, input_1=None, input_2=None, input_3=None, 
                         input_4=None, input_5=None):
        lazy_inputs = []
        
        # Check which inputs need to be evaluated
        if switch_condition == case_1:
            if input_1 is None:
                lazy_inputs.append("input_1")
        elif switch_condition == case_2:
            if input_2 is None:
                lazy_inputs.append("input_2")
        elif switch_condition == case_3:
            if input_3 is None:
                lazy_inputs.append("input_3")
        elif switch_condition == case_4:
            if input_4 is None:
                lazy_inputs.append("input_4")
        elif switch_condition == case_5:
            if input_5 is None:
                lazy_inputs.append("input_5")
        else:
            if input_default is None:
                lazy_inputs.append("input_default")
                
        return lazy_inputs

    def switch_case(self, switch_condition, case_1, case_2, case_3, case_4, case_5, 
                   input_default, input_1=None, input_2=None, input_3=None, 
                   input_4=None, input_5=None):
        output = input_default
        
        if switch_condition == case_1 and input_1 is not None:
            output = input_1
        elif switch_condition == case_2 and input_2 is not None:
            output = input_2
        elif switch_condition == case_3 and input_3 is not None:
            output = input_3
        elif switch_condition == case_4 and input_4 is not None:
            output = input_4
        elif switch_condition == case_5 and input_5 is not None:
            output = input_5
            
        return (output,)