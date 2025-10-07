import ast
import operator as op

class FL_Math:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "A": ("FLOAT", {"default": 0, "min": -1000000, "max": 1000000, "step": 0.01}),
                "B": ("FLOAT", {"default": 0, "min": -1000000, "max": 1000000, "step": 0.01}),
                "C": ("FLOAT", {"default": 0, "min": -1000000, "max": 1000000, "step": 0.01}),
                "equation": ("STRING", {"multiline": True, "default": "A + B * C"}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT")
    FUNCTION = "calculate"
    CATEGORY = "🏵️Fill Nodes/Utility"

    def calculate(self, A, B, C, equation):
        # Define allowed operators
        operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                     ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

        def eval_expr(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_expr(node.operand))
            elif isinstance(node, ast.Name):
                return {"A": A, "B": B, "C": C}[node.id]
            else:
                raise TypeError(node)

        try:
            result = eval_expr(ast.parse(equation, mode='eval').body)
            return (int(result), float(result))
        except Exception as e:
            return (0, 0.0)  # Return default values in case of error