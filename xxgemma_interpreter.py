import sys
import math
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Set

# ==========================================
# 1. 依存ライブラリのダミー/フォールバック処理
# ==========================================
try:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

    class StoppingCriteria:
        pass

    class StoppingCriteriaList(list):
        def __init__(self, items=None):
            super().__init__(items or [])

    class TextStreamer:
        def __init__(self, tokenizer, skip_prompt=False):
            self.tokenizer = tokenizer
            self.skip_prompt = skip_prompt

        def put(self, value):
            pass

        def end(self):
            pass


# ==========================================
# 1.2 リアルLLM環境用 StoppingCriteria (改行での一時停止制御)
# ==========================================
if HAS_TRANSFORMERS:
    class StopOnNewLineCriteria(StoppingCriteria):
        """リアルLLM環境で改行(\\n)が出力された時点で生成を一時停止させるStoppingCriteria"""
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            if input_ids.shape[1] == 0:
                return False
            last_token_id = input_ids[0, -1].item()
            decoded = self.tokenizer.decode([last_token_id])
            if "\n" in decoded or "\r" in decoded:
                return True
            return False
else:
    class StopOnNewLineCriteria:
        def __init__(self, tokenizer):
            pass
        def __call__(self, input_ids, scores, **kwargs):
            return False


# ==========================================
# 2. ロボットシミュレータ (robot1 のハードコーディング)
# ==========================================
ROBOT1_MODE = "normal"
#ROBOT1_MODE = "recoverable"
#ROBOT1_MODE = "fatal"
#ROBOT1_MODE = "battery"

class Robot1Exception(RuntimeError):
    pass

class DummyRobot1:
    """DSL内で import robot1 した際に利用可能になる関数群"""
    def __init__(self, mode="normal"):
        self.scan_count = 0
        self.mode = mode

    def robot1_scan(self) -> str:
        self.scan_count += 1
        return f"robot1_tensor_scan_data_{self.scan_count:03d}"

    def robot1_is_normal(self, tensor_data: str) -> bool:
        return True #self.mode == "normal"

    def robot1_model(self, tensor_data: str) -> str:
        return f"robot1_tensor_processed_model_for_{tensor_data}"

    def robot1_act(self, tensor_model: str) -> str:
        print(f"🤖 [ROBOT ACTION] {tensor_model} に基づいてロボットが作動しました。")
        self.robot1_normal_act()
        return "success"

    def robot1_normal_act(self) -> str:
        if self.mode == "normal":
            return "success"

        elif self.mode == "recoverable":
            raise Robot1Exception("Robot lost balance.")

        elif self.mode == "fatal":
            raise Robot1Exception("Motor controller failure.")

        else:
            raise Robot1Exception(f"Unknown robot mode: {self.mode}")

    def robot1_error_router_determine(self) -> str:
        if self.mode == "recoverable":
            return "walking"

        elif self.mode == "battery":
            return "maintenance_dock"

        elif self.mode == "fatal":
            return "unhandlable"

        else:
            return "normal"

    def robot1_error_router_reinforce(self, loss) -> None:
        return None

    def robot1_walk_error_recovery(self) -> str:
        return "robot1_tensor_walk_recovery_completed"

    def robot1_battery_error_recovery(self):
        return "robot1_tensor_battery_recovery_completed"

    def robot1_unhandlable_error(self) -> str:
        return "robot1_unhandlable_error_detected"


# ==========================================
# 2.5 DSL用補助パース関数
# ==========================================
def split_dsl_elements(s: str) -> List[str]:
    """
    引用符や括弧によるネストを保護しつつ、
    トップレベルにカンマがあればカンマで、なければスペースで分割する。
    """
    s = s.strip()
    if not s:
        return []
        
    keywords = {'str', 'int', 'float', 'bool', 'list', 'dict', 'tensor'}
    
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    in_quote = None
    
    # トップレベルの文字だけを残し、他をスペースで埋めた文字列を作る（キーワード検出用）
    top_level_chars = []
    for char in s:
        if char in ('"', "'"):
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
            top_level_chars.append(' ')
        elif in_quote is not None:
            top_level_chars.append(' ')
        elif char == '(':
            paren_depth += 1
            top_level_chars.append(' ')
        elif char == ')':
            paren_depth -= 1
            top_level_chars.append(' ')
        elif char == '[':
            bracket_depth += 1
            top_level_chars.append(' ')
        elif char == ']':
            bracket_depth -= 1
            top_level_chars.append(' ')
        elif char == '{':
            brace_depth += 1
            top_level_chars.append(' ')
        elif char == '}':
            brace_depth -= 1
            top_level_chars.append(' ')
        else:
            if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                top_level_chars.append(char)
            else:
                top_level_chars.append(' ')
                
    top_level_str = "".join(top_level_chars)
    
    # トップレベルにあるキャストキーワードの開始インデックスを特定
    kw_indices = []
    for match in re.finditer(r'\b(str|int|float|bool|list|dict|tensor)\b', top_level_str):
        kw_indices.append(match.start())
        
    num_kws = len(kw_indices)
    
    # トップレベルにカンマがあるか
    has_top_comma = False
    for char in top_level_str:
        if char == ',':
            has_top_comma = True
            break
            
    split_char = ',' if has_top_comma else ' '
    
    parts = []
    current = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    in_quote = None
    
    # 分割判定用の正規表現（辞書のキー前置 key: を許容したキャストキーワード開始パターン）
    starts_with_kw_pattern = r'^([a-zA-Z0-9_]+:)?\s*(str|int|float|bool|list|dict|tensor)\b'
    
    for idx, char in enumerate(s):
        if char in ('"', "'"):
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
            current.append(char)
        elif in_quote is not None:
            current.append(char)
        elif char == '(':
            paren_depth += 1
            current.append(char)
        elif char == ')':
            paren_depth -= 1
            current.append(char)
        elif char == '[':
            bracket_depth += 1
            current.append(char)
        elif char == ']':
            bracket_depth -= 1
            current.append(char)
        elif char == '{':
            brace_depth += 1
            current.append(char)
        elif char == '}':
            brace_depth -= 1
            current.append(char)
        elif char == split_char and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            should_split = True
            if split_char == ',' and num_kws >= 2:
                # キーワードが複数含まれる場合、このカンマの右側が新しいキャスト式で始まっている場合のみ分割
                right_sub = s[idx+1:].strip()
                if not re.match(starts_with_kw_pattern, right_sub):
                    should_split = False
                    
            if should_split:
                val = "".join(current).strip()
                if val:
                    parts.append(val)
                current = []
            else:
                current.append(char)
        else:
            current.append(char)
            
    val = "".join(current).strip()
    if val:
        parts.append(val)
    return parts


# ==========================================
# 3. DSL インタプリタ
# ==========================================
class InterpreterAbort(Exception):
    def __init__(self, message):
        self.message = message

class xxGemmaInterpreter:
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        # tuple (content_string, is_newline_bound) としてSTATEMENTブロック/プレーンテキストを格納
        self.statement_parts: List[Tuple[str, bool]] = []  
        self.last_result: Any = None
        self.imported_modules: Set[str] = set()
        self.returned_value: Any = None
        self.exception: Optional[str] = None
        self.last_feedback_type: Optional[str] = None
        self.robot = DummyRobot1(mode=ROBOT1_MODE)
        self.last_line_type: Optional[str] = None # 'statement', 'code', 'comment', 'result', 'exception'

    def clean_var_name(self, name: str) -> str:
        """変数名から先頭 of $ を取り除く"""
        name = name.strip()
        if name.startswith("$"):
            return name[1:]
        return name

    def escape_feedback(self, text: str) -> str:
        """RESULTやEXCEPTIONに含まれる改行やタブをエスケープして1行にする"""
        if not isinstance(text, str):
            text = str(text)
        # まずバックスラッシュ自体をエスケープする
        text = text.replace('\\', '\\\\')
        # その後、本物の改行、キャリッジリターン、タブをエスケープ文字に置換
        text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return text

    def _evaluate_dot_chain(self, var_name: str, chain_str: str) -> Any:
        """$d.user.name や $l.2, $d.'animal' のようなドットチェーンを辿って値を解決する"""
        if var_name not in self.variables:
            raise NameError(f"Variable '{var_name}' is not defined.")
        curr = self.variables[var_name]
        if not chain_str:
            return curr
        
        # アルファベット・数字、およびシングル/ダブルクォーテーションで囲まれた文字列を切り出して走査
        parts = re.findall(r'\.([a-zA-Z0-9_]+|\'[^\']*\'|"[^"]*")', chain_str)
        for part in parts:
            if (part.startswith("'") and part.endswith("'")) or (part.startswith('"') and part.endswith('"')):
                part = part[1:-1]
                
            if isinstance(curr, dict):
                if part in curr:
                    curr = curr[part]
                else:
                    raise KeyError(f"Key '{part}' not found in dict.")
            elif isinstance(curr, (list, tuple)):
                try:
                    idx = int(part)
                    curr = curr[idx]
                except ValueError:
                    raise TypeError(f"List index must be integer, got '{part}'.")
                except IndexError:
                    raise IndexError(f"List index {idx} out of range.")
            else:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    raise AttributeError(f"Object has no attribute '{part}'.")
        return curr

    def evaluate_expr(self, expr: str, strict: bool = False) -> Any:
        """DSL内の式を評価する (安全な eval)"""
        expr = expr.strip().rstrip(';')
        
        # 特殊変数 $RESULT の展開
        if "$RESULT" in expr:
            val_str = repr(self.last_result) if self.last_result is not None else "None"
            expr = expr.replace("$RESULT", val_str)

        # 特殊変数 $STATEMENT の展開
        if "$STATEMENT" in expr:
            stmt_val = self.get_statement()
            expr = expr.replace("$STATEMENT", repr(stmt_val))

        # --- 1. DSL型キャスト (変数置換より前に評価することで、生の構造を保ったまま再帰パースする) ---
        type_match = re.match(r'^(str|int|float|bool|list|dict|tensor)\s+(.*)$', expr)
        if type_match:
            cast_type = type_match.group(1)
            inner_expr = type_match.group(2).strip()
            
            if cast_type == "str":
                val = self.evaluate_expr(inner_expr)
                if isinstance(val, str):
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                return str(val)
            elif cast_type == "int":
                return int(self.evaluate_expr(inner_expr))
            elif cast_type == "float":
                return float(self.evaluate_expr(inner_expr))
            elif cast_type == "bool":
                val = self.evaluate_expr(inner_expr)
                if isinstance(val, str):
                    return val.strip().lower() in ("true", "t", "1", "yes", "y")
                return bool(val)
            elif cast_type == "list":
                elements = split_dsl_elements(inner_expr)
                return [self.evaluate_expr(e) for e in elements]
            elif cast_type == "dict":
                elements = split_dsl_elements(inner_expr)
                d_val = {}
                for item in elements:
                    if ":" in item:
                        k, v = item.split(":", 1)
                        k_str = str(self.evaluate_expr(k.strip()))
                        d_val[k_str] = self.evaluate_expr(v.strip())
                    else:
                        clean_item = self.clean_var_name(item)
                        if clean_item in self.variables:
                            d_val[clean_item] = self.variables[clean_item]
                        else:
                            d_val[clean_item] = self.evaluate_expr(item)
                return d_val
            elif cast_type == "tensor":
                return self.evaluate_expr(inner_expr)

        # 2. 動的インデックス表現 .(expression) を内側から順に再帰的に評価・置換する
        def replace_dynamic(match):
            expr_inside = match.group(1)
            evaluated = self.evaluate_expr(expr_inside)
            return f".{repr(evaluated)}"
        
        while True:
            next_expr, count = re.subn(r'\.\(([^()]+)\)', replace_dynamic, expr)
            if count == 0:
                break
            expr = next_expr

        # 3. 変数およびドットチェーンの置換解決 ($var.key1.key2、およびシングル/ダブルクォーテーション付きを含む)
        def replace_chain(match):
            var_name = match.group(1)
            chain = match.group(2)
            try:
                if var_name == "RESULT":
                    if chain:
                        curr = self.last_result
                        parts = re.findall(r'\.([a-zA-Z0-9_]+|\'[^\']*\'|"[^"]*")', chain)
                        for part in parts:
                            if (part.startswith("'") and part.endswith("'")) or (part.startswith('"') and part.endswith('"')):
                                part = part[1:-1]
                            if isinstance(curr, dict):
                                curr = curr[part]
                            elif isinstance(curr, (list, tuple)):
                                curr = curr[int(part)]
                            else:
                                curr = getattr(curr, part)
                        val = curr
                    else:
                        val = self.last_result
                else:
                    val = self._evaluate_dot_chain(var_name, chain)

                # 式の評価用途として Python 構文に正しくマッピングするため、常に repr(val) で引用符付き展開する
                return repr(val)
            except Exception as e:
                raise e

        # 計算式、関数呼び出し、四則演算が含まれているか確認
        is_function_or_calc = "(" in expr or ")" in expr or any(op in expr for op in ["+", "-", "*", "/"])

        try:
            # 展開対象となるドットチェーンのパターンを定義（文字列リテラルドットを含む）
            pattern_chain = r'\$([a-zA-Z0-9_]+)((?:\.(?:[a-zA-Z0-9_]+|\'[^\']*\'|"[^"]*"))*)'
            expr = re.sub(pattern_chain, replace_chain, expr)

            eval_globals = {
                "__builtins__": None,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "pi": math.pi,
                "true": True,
                "false": False,
                "True": True,
                "False": False,
            }

            if "robot1" in self.imported_modules:
                eval_globals.update({
                    "robot1_scan": self.robot.robot1_scan,
                    "robot1_is_normal": self.robot.robot1_is_normal,
                    "robot1_model": self.robot.robot1_model,
                    "robot1_act": self.robot.robot1_act,
                    "robot1_normal_act": self.robot.robot1_normal_act,
                    "robot1_error_router_determine": self.robot.robot1_error_router_determine,
                    "robot1_error_router_reinforce": self.robot.robot1_error_router_reinforce,
                    "robot1_walk_error_recovery": self.robot.robot1_walk_error_recovery,
                    "robot1_battery_error_recovery": self.robot.robot1_battery_error_recovery,
                    "robot1_unhandlable_error": self.robot.robot1_unhandlable_error,
                })

            eval_locals = {}
            # DSLの変数解決は事前に $ 置換によって完全に展開されるため、
            # 直接 eval に変数テーブルを流し込まない（生のベアワードと衝突するのを防ぐため空とする）

            # 厳格モード(strict)時の未定義の識別子/タイポチェック
            if strict:
                temp_expr = expr
                temp_expr = re.sub(r"'[^']*'", "", temp_expr)
                temp_expr = re.sub(r'"[^"]*"', "", temp_expr)
                identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', temp_expr)
                allowed_identifiers = set(eval_globals.keys()) | set(eval_locals.keys()) | {
                    "True", "False", "None", "true", "false",
                    "str", "int", "float", "bool", "list", "dict", "tensor"
                }
                for ident in identifiers:
                    if ident not in allowed_identifiers:
                        raise NameError(f"Name '{ident}' is not defined.")

            if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
                try:
                    # 安全な空コンテキストでevalを通すことでエスケープシーケンスを安全に解決する
                    return eval(expr, {"__builtins__": None}, {})
                except Exception:
                    return expr[1:-1]
            return eval(expr, eval_globals, eval_locals)
        except Exception as e:
            if is_function_or_calc or strict:
                raise e
            if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
                return expr[1:-1]
            return expr

    def execute_line(self, line: str) -> Optional[str]:
        """DSLの1行を実行する。"""
        # 特殊制御トークンのクレンジング（パース崩壊を防止）
        line = re.sub(r'<\|turn\>|<turn\|>|<eos>|<bos>', '', line)
        line = line.strip()
        if not line:
            return None

        # [COMMENT]
        if line.startswith("[COMMENT]"):
            self.last_line_type = "comment"
            return None

        # [RESULT]
        if line.startswith("[RESULT]"):
            self.last_line_type = "result"
            return None

        # [EXCEPTION]
        if line.startswith("[EXCEPTION]"):
            exc_content = line[len("[EXCEPTION]"):].strip()
            self.exception = self.escape_feedback(exc_content)
            self.last_line_type = "exception"
            return None

        # [STATEMENT]
        if line.startswith("[STATEMENT]"):
            stmt_content = line[len("[STATEMENT]"):].strip()
            # 直前が STATEMENT 系の行であれば改行で結合するようにフラグを立てる
            is_newline_bound = (self.last_line_type == "statement")
            self.statement_parts.append((stmt_content, is_newline_bound))
            self.last_line_type = "statement"
            return None

        # [CODE]
        if line.startswith("[CODE]"):
            code_content = line[len("[CODE]"):].strip()
            self.last_line_type = "code"
            try:
                feedback = self._execute_code(code_content)
                self.exception = None
                return feedback
            except InterpreterAbort:
                raise
            except Exception as e:
                err_msg = self.escape_feedback(f"{type(e).__name__}: {str(e)}")
                self.exception = err_msg
                return f"[EXCEPTION]{err_msg}"

        # プレーンテキスト
        processed_text = line
        is_newline_bound = (self.last_line_type == "statement")
        self.statement_parts.append((processed_text, is_newline_bound))
        self.last_line_type = "statement"
        return None

    def _execute_code(self, code: str) -> Optional[str]:
        # 末尾のセミコロンを事前に除去
        code = code.strip().rstrip(';')
        if not code:
            return None

        # --- raise 文 ---
        if code.startswith("raise "):
            err_msg = code[6:].strip()
            if (err_msg.startswith('"') and err_msg.endswith('"')) or (err_msg.startswith("'") and err_msg.endswith("'")):
                err_msg = err_msg[1:-1].strip()
            else:
                err_msg = re.sub(r'([a-zA-Z0-9_]+):\s*["\'](.*?)["\']', r'\1: \2', err_msg)
            raise InterpreterAbort(self.escape_feedback(err_msg))

        # --- return 文 ---
        if code.startswith("return "):
            expr = code[7:].strip()
            self.returned_value = self.evaluate_expr(expr, strict=True)
            return None

        # --- import 文 ---
        if code.startswith("import "):
            module = code[7:].strip()
            self.imported_modules.add(module)
            return None

        # --- show 文 / print 文 (等価にサポート) ---
        if code.startswith("show ") or code.startswith("print "):
            prefix_len = 5 if code.startswith("show ") else 6
            expr = code[prefix_len:].strip()
            
            clean_expr = self.clean_var_name(expr)
            if clean_expr in self.variables:
                val = self.variables[clean_expr]
            else:
                val = self.evaluate_expr(expr, strict=True)
            self.last_result = val
            
            if isinstance(val, (dict, list)):
                escaped_val = self.escape_feedback(json.dumps(val, ensure_ascii=False))
            elif isinstance(val, bool):
                escaped_val = str(val).lower()
            else:
                escaped_val = self.escape_feedback(str(val))
            return f"[RESULT]{escaped_val}"

        # --- amend 文 ---
        if code.startswith("amend "):
            amend_content = code[6:].strip()
            return self._execute_amend(amend_content)

        # --- del 文 ---
        if code.startswith("del "):
            del_content = code[4:].strip()
            self._execute_del(del_content)
            return None

        # --- 代入文 ---
        if "=" in code:
            parts = code.split("=", 1)
            var_part = parts[0].strip()
            val_part = parts[1].strip()

            var_name = self.clean_var_name(var_part)
            val = self.evaluate_expr(val_part, strict=True)
            self.variables[var_name] = val
            return None

        # キャスト・数式評価をすべて evaluate_expr に一本化
        val = self.evaluate_expr(code, strict=True)
        self.last_result = val
        return f"[RESULT]{self.escape_feedback(str(val))}"

    def _execute_amend(self, amend_str: str) -> Optional[str]:
        if "->" not in amend_str:
            raise SyntaxError("Invalid amend syntax.")
        before_part, after_part = amend_str.split("->", 1)
        after_str = after_part.strip()
        before_part = before_part.strip()

        if ":" in before_part:
            target_part, search_part = before_part.split(":", 1)
            target = target_part.strip()
            search_str = search_part.strip()
            search_str_resolved = str(self.evaluate_expr(search_str))
            after_str_resolved = str(self.evaluate_expr(after_str))
            if search_str_resolved == "":
                raise ValueError("amend target cannot be empty.")

            if target == "$STATEMENT":
                current_statement = self.get_statement()
                if search_str_resolved not in current_statement:
                    raise ValueError(f"amend target not found: {search_str_resolved}")
                new_statement = current_statement.replace(search_str_resolved, after_str_resolved)
                # statement_parts を一度フラットにして置き換え
                self.statement_parts = [(new_statement, False)]
            else:
                var_name = self.clean_var_name(target)
                if var_name in self.variables:
                    orig_val = str(self.variables[var_name])
                    if search_str_resolved not in orig_val:
                        raise ValueError(f"amend target not found: {search_str_resolved}")
                    new_val = orig_val.replace(search_str_resolved, after_str_resolved)
                    if isinstance(self.variables[var_name], bool):
                        self.variables[var_name] = (
                            new_val.lower() in ("true", "1", "yes")
                        )
                    elif isinstance(self.variables[var_name], int):
                        try: self.variables[var_name] = int(new_val)
                        except: self.variables[var_name] = new_val
                    elif isinstance(self.variables[var_name], float):
                        try: self.variables[var_name] = float(new_val)
                        except: self.variables[var_name] = new_val
                    else:
                        self.variables[var_name] = new_val
                else:
                    raise NameError(f"Variable '{var_name}' is not defined.")
        else:
            search_str_resolved = str(self.evaluate_expr(before_part))
            after_str_resolved = str(self.evaluate_expr(after_str))
            if search_str_resolved == "":
                raise ValueError("amend target cannot be empty.")
            current_statement = self.get_statement()
            if search_str_resolved not in current_statement:
                raise ValueError(f"amend target not found: {search_str_resolved}")
            new_statement = current_statement.replace(search_str_resolved, after_str_resolved)

            self.statement_parts = [(new_statement, False)]

        return None

    def _execute_del(self, del_str: str):
        del_str = del_str.strip()
        if " from " in del_str:
            target_part, dict_part = del_str.split(" from ", 1)
            target_key = self.clean_var_name(target_part.strip())
            dict_name = self.clean_var_name(dict_part.strip())
            if dict_name in self.variables and isinstance(self.variables[dict_name], dict):
                if target_key in self.variables[dict_name]:
                    del self.variables[dict_name][target_key]
        else:
            # 単純な変数削除、または $d.user.age のようなドット指定削除に対応
            var_name = self.clean_var_name(del_str)
            if "." in var_name:
                parts = var_name.split('.')
                root_var = parts[0]
                if root_var in self.variables:
                    curr = self.variables[root_var]
                    # 最後の手前の階層まで辿る
                    for part in parts[1:-1]:
                        if isinstance(curr, dict) and part in curr:
                            curr = curr[part]
                        elif isinstance(curr, list):
                            try:
                                curr = curr[int(part)]
                            except:
                                break
                        else:
                            break
                    last_part = parts[-1]
                    # 最深部でキーまたはインデックスを削除
                    if isinstance(curr, dict) and last_part in curr:
                        del curr[last_part]
                    elif isinstance(curr, list):
                        try:
                            idx = int(last_part)
                            del curr[idx]
                        except:
                            pass
            else:
                if var_name in self.variables:
                    del self.variables[var_name]

    def _replace_var_in_text(self, match) -> str:
        var_name = match.group(1)
        chain = match.group(2)
        try:
            # RESULT の場合は last_result を起点に置換解決
            if var_name == "RESULT":
                if chain:
                    curr = self.last_result
                    parts = chain.lstrip('.').split('.')
                    for part in parts:
                        if not part:
                            continue
                        if isinstance(curr, dict):
                            curr = curr[part]
                        elif isinstance(curr, (list, tuple)):
                            curr = curr[int(part)]
                        else:
                            curr = getattr(curr, part)
                    return str(curr)
                else:
                    return str(self.last_result) if self.last_result is not None else match.group(0)
            
            val = self._evaluate_dot_chain(var_name, chain)
            return str(val)
        except Exception:
            return match.group(0)

    def get_statement(self) -> str:
        """蓄積された文（__STATEMENT__）を結合し、最終的な変数置換を行って返す"""
        if not self.statement_parts:
            return ""
        
        chunks = []
        for content, is_newline_bound in self.statement_parts:
            content_str = content.replace(r"\(", "__L_PAREN__").replace(r"\)", "__R_PAREN__")
            
            # 1. 蓄積テキスト内でも動的括弧表現 .(expression) を再帰評価
            while True:
                def replace_dyn_stmt(m):
                    inside = m.group(1)
                    return f".{repr(self.evaluate_expr(inside))}"
                next_str, count = re.subn(r'\.\(([^()]+)\)', replace_dyn_stmt, content_str)
                if count == 0:
                    break
                content_str = next_str

            # 2. 変数およびドットチェーンの置換解決 ($var.key1.key2)
            content_str = re.sub(r'\$([a-zA-Z0-9_]+)((?:\.[a-zA-Z0-9_]+)*)', self._replace_var_in_text, content_str)
            content_str = content_str.replace("__L_PAREN__", "(").replace("__R_PAREN__", ")")
            content_str = re.sub(r'\\([\\(){}\[\]$#!?.,:;+\-*="\'])', r'\1', content_str)
            content_str = content_str.strip()
            
            if not content_str:
                continue
                
            if not chunks:
                chunks.append(content_str)
            else:
                if is_newline_bound:
                    chunks.append("\n" + content_str)
                else:
                    chunks.append(" " + content_str)
                    
        full_stmt = "".join(chunks)
        
        # 連続した余分なスペースをまとめつつ、改行構造を維持する
        normalized_lines = []
        for line in full_stmt.split('\n'):
            line_cleaned = re.sub(r'[ \t]+', ' ', line).strip()
            if line_cleaned:
                normalized_lines.append(line_cleaned)
        return "\n".join(normalized_lines)

    def to_json(self) -> Dict[str, Any]:
        """インタプリタの状態を JSON 辞書として出力する"""
        result_dict = {}
        for k, v in self.variables.items():
            result_dict[k] = v

        result_dict["__STATEMENT__"] = self.get_statement()

        if self.returned_value is not None:
            result_dict["__RETURN__"] = self.returned_value
        if self.exception is not None:
            result_dict["__ERROR__"] = self.exception

        return result_dict


# ==========================================
# 4. テスト用ダミー LLM モデルとトークナイザ
# ==========================================
class DummyTensor:
    def __init__(self, token_ids: List[int], text: str = ""):
        self.token_ids = token_ids
        self.text = text

    def to(self, device):
        return self

    @property
    def shape(self):
        return (1, len(self.token_ids))

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            row, col = item
            if row == 0 and isinstance(col, slice):
                sliced_ids = self.token_ids[col]
                return DummyTensor(sliced_ids, text=self.text)
        return self


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 999

    def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, tokenize: bool = True, return_dict: bool = True, **kwargs) -> Dict[str, Any]:
        full_text = ""
        for msg in messages:
            role = msg.get("role", "")
            content_list = msg.get("content", [])
            
            text_content = ""
            if isinstance(content_list, list):
                for part in content_list:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_content += part.get("text", "")
                    elif isinstance(part, str):
                        text_content += part
            elif isinstance(content_list, str):
                text_content = content_list
                
            full_text += f"<|turn>{role}\n{text_content}<turn|>\n"
            
        dummy_ids = [ord(c) for c in full_text[:10]]
        return {"input_ids": DummyTensor(dummy_ids, text=full_text)}

    def decode(self, token_tensor, skip_special_tokens=False, **kwargs) -> str:
        if isinstance(token_tensor, DummyTensor):
            return token_tensor.text
        if isinstance(token_tensor, list):
            return "".join([chr(x) for x in token_tensor if x != self.eos_token_id])
        return ""

    def encode(self, text: str, add_special_tokens=False, return_tensors=None, **kwargs):
        dummy_ids = [ord(c) for c in text]
        if return_tensors == "pt":
            return DummyTensor(dummy_ids, text=text)
        return dummy_ids


class DummyModel:
    def __init__(self, scenario_text: str):
        self.lines = [line for line in scenario_text.split("\n")]
        self.current_line_idx = 0

    def generate(self, input_ids: DummyTensor, stopping_criteria=None, **kwargs) -> DummyTensor:
        if self.current_line_idx < len(self.lines):
            next_line = self.lines[self.current_line_idx]
            self.current_line_idx += 1
        else:
            next_line = ""

        is_last = self.current_line_idx >= len(self.lines)
        token_ids = [1, 2, 3]
        if is_last:
            token_ids.append(999)

        if "streamer" in kwargs and kwargs["streamer"]:
            streamer = kwargs["streamer"]
            print(f"{next_line}")
            
        return DummyTensor(token_ids, text=next_line)


# ==========================================
# 5. インタプリタ実行メインループ
# ==========================================
def do_gemma_4_line_by_line_inference(
    messages: List[Dict[str, Any]], 
    tokenizer: Any, 
    model: Any, 
    max_lines: int = 25
) -> Tuple[Dict[str, Any], str]:
    """DSLインタプリタを統合した、Gemma 4 E2B準拠の行ごと履歴推論ループ"""
    
    interpreter = xxGemmaInterpreter()
    generated_lines: List[str] = []

    has_model_role = any(m.get("role") == "model" for m in messages)
    if not has_model_role:
        messages.append({"role": "model", "content": []})
        
    model_msg_idx = next(i for i, m in enumerate(messages) if m.get("role") == "model")

    use_real_transformers = HAS_TRANSFORMERS and not isinstance(model, DummyModel)

    stop_criteria = None
    if use_real_transformers:
        stop_criteria = StoppingCriteriaList([StopOnNewLineCriteria(tokenizer)])

    print("=== 🤖 xxLLM 行ごと生成ループ開始 ===")

    abort = False
    
    for line_idx in range(max_lines):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt" if use_real_transformers else None,
        )
        
        if isinstance(encoded, dict):
            encoded.pop("mm_token_type_ids", None)
            
        if use_real_transformers:
            input_ids = encoded["input_ids"]
            gen_inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in encoded.items()}
        else:
            input_ids = encoded["input_ids"]
            gen_inputs = {"input_ids": input_ids}

        if use_real_transformers:
            outputs = model.generate(
                **gen_inputs,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stop_criteria,
                streamer=TextStreamer(tokenizer, skip_prompt=True)
            )
            input_len = gen_inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_len:]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        else:
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                streamer=TextStreamer(tokenizer, skip_prompt=True)
            )
            new_text = tokenizer.decode(outputs, skip_special_tokens=True)

        clean_text = new_text.strip("\r\n")
        
        # 特殊トークンを事前に除去
        clean_text = re.sub(r'<\|turn\>|<turn\|>|<eos>|<bos>', '', clean_text)
        
        # [CODE] などを目印に、同一物理行に混在しているタグを個々の命令行に切り分ける
        pattern = r'(\[CODE\]|\[STATEMENT\]|\[COMMENT\]|\[EXCEPTION\]|\[RESULT\])'
        splits = re.split(pattern, clean_text)
        
        raw_lines = []
        current_tag = ""
        
        for part in splits:
            if not part:
                continue
            if re.match(pattern, part):
                current_tag = part
            else:
                if current_tag:
                    raw_lines.append(f"{current_tag} {part.strip()}")
                    current_tag = ""
                else:
                    if part.strip():
                        raw_lines.append(part.strip())

        if not raw_lines:
            if line_idx > 0:
                break
            else:
                raw_lines = [""]

        for clean_line in raw_lines:
            clean_line = clean_line.strip()
            if not clean_line:
                continue

            # --------------------------------------------------
            # ★ AIが自分で [RESULT] などのシステムタグを出力してしまった場合のガード
            # --------------------------------------------------
            if clean_line.startswith("[RESULT]"):
                print(f"⚠️ [SYSTEM GUARD]: AIがRESULTを自己生成したため無視します: {clean_line}")
                interpreter.last_feedback_type = None
                continue
            if clean_line.startswith("[EXCEPTION]"):

                if interpreter.last_feedback_type == "EXCEPTION":
                    print(
                        f"⚠️ [SYSTEM GUARD]: EXCEPTIONのエコーバックを検出したため無視します: {clean_line}"
                    )
                    interpreter.last_feedback_type = None
                    continue
                if interpreter.last_feedback_type == "RESULT":
                    print(
                        f"⚠️ [SYSTEM GUARD]: RESULTフィードバック直後のEXCEPTIONを無視します: {clean_line}"
                    )
                    interpreter.last_feedback_type = None
                    continue
                print(f"⚠️ [SYSTEM GUARD]: AIが勝手にEXCEPTIONを生成しました: {clean_line}")
                feedback_result = "[SYSTEM]EXCEPTION_ALREADY_HANDLED"
                print(f"      [SYSTEM FEEDBACK TO LLM]: {feedback_result}")
                interpreter.last_feedback_type = "SYSTEM"
                generated_lines.append(feedback_result)
                messages[model_msg_idx]["content"].append({
                    "type": "text",
                    "text": feedback_result + "\n"
                })
                continue

            interpreter.last_feedback_type = None
            
            generated_lines.append(clean_line)

            messages[model_msg_idx]["content"].append({
                "type": "text",
                "text": clean_line + "\n"
            })

            print(f"← 🛠️ [Python制御: {line_idx + 1}行目の生成終了を検知しました]")

            try:
                feedback_result = interpreter.execute_line(clean_line)
            except InterpreterAbort as e:
                interpreter.exception = e.message
                print(f"    ↳ ★ DSL raise により終了: {e.message}")
                abort = True
                break

            if feedback_result:
                print(f"    ↳ ★インタプリタによる結果を検出: {feedback_result}")
                print(f"      [SYSTEM FEEDBACK TO LLM]: {feedback_result}")

                if feedback_result.startswith("[RESULT]"):
                    interpreter.last_feedback_type = "RESULT"
                elif feedback_result.startswith("[EXCEPTION]"):
                    interpreter.last_feedback_type = "EXCEPTION"
                else:
                    interpreter.last_feedback_type = None

                generated_lines.append(feedback_result)
                
                messages[model_msg_idx]["content"].append({
                    "type": "text",
                    "text": feedback_result + "\n"
                })

        if abort:
            break

        # EOSの検知
        if use_real_transformers:
            if outputs[0][-1].item() == tokenizer.eos_token_id:
                print("=== 🔚 EOS（終了トークン）を検知したため全体を終了します ===")
                break
        else:
            if isinstance(outputs, DummyTensor) and 999 in outputs.token_ids:
                print("=== 🔚 EOS（終了トークン）を検知したため全体を終了します ===")
                break

    print("=== ⚙️ インタプリタ実行完了後の内部状態 ===")
    final_json = interpreter.to_json()
    print(json.dumps(final_json, indent=2, ensure_ascii=False))
    
    full_program_text = "\n".join(generated_lines)
    return final_json, full_program_text


# ==========================================
# 6. テスト自動検証スイート
# ==========================================
def run_tests():
    print("\n" + "="*50)
    print("🧪 xxGemma DSL インタプリタ テスト開始 (Gemma 4 E2B準拠)")
    print("="*50)

    # --------------------------------------------------
    # テストケース 1: Cat & Balls (基本 & 型キャスト & show)
    # --------------------------------------------------
    print("\n--- [Test Case 1] Cat and Balls ---")
    
    scenario_1 = """[CODE]animal = str long cat
The $animal
[CODE]int 1 + 1
[RESULT]2
[CODE]balls = int $RESULT
plays $balls balls."""

    dummy_tokenizer = DummyTokenizer()
    dummy_model_1 = DummyModel(scenario_1)

    messages_1 = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI interacting with the xxGemma interpreter.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Generate text that long cat plays two balls.",
                }
            ],
        }
    ]

    result_json, dsl_prog_1 = do_gemma_4_line_by_line_inference(messages_1, dummy_tokenizer, dummy_model_1)
    
    print("\n--- [Generated DSL Program] ---")
    print(dsl_prog_1)
    print("-------------------------------")
    
    assert result_json.get("animal") == "long cat", "Test1 Failed: animal"
    assert result_json.get("balls") == 2, "Test1 Failed: balls"
    assert result_json.get("__STATEMENT__") == "The long cat plays 2 balls.", f"Test1 Failed: statement -> {result_json.get('__STATEMENT__')}"
    print("✅ Test Case 1: PASS")


    # --------------------------------------------------
    # テストケース 2: ベアワード回避, 三角関数, エスケープ, del, bool型評価
    # --------------------------------------------------
    print("\n--- [Test Case 2] Advanced Features (incl. Bool evaluation) ---")
    
    scenario_2 = """[CODE]animal = str "long cat"
The $animal
[CODE]balls = int 1 + cos(0)
[CODE]print balls
[RESULT]2
plays $RESULT ball\\(s\\).
[CODE]is_cat = bool true
[CODE]d = dict $animal, b:$balls, c:$is_cat
[CODE]l = list $animal, $balls, $is_cat
[CODE]del balls
[CODE]del animal from d"""

    dummy_model_2 = DummyModel(scenario_2)
    messages_2 = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Interact with interpretation system."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Execute complex statements."}],
        }
    ]
    result_json_2, dsl_prog_2 = do_gemma_4_line_by_line_inference(messages_2, dummy_tokenizer, dummy_model_2)

    print("\n--- [Generated DSL Program] ---")
    print(dsl_prog_2)
    print("-------------------------------")

    assert "balls" not in result_json_2, "Test2 Failed: del balls"
    assert result_json_2.get("is_cat") is True, f"Test2 Failed: bool assignment -> {result_json_2.get('is_cat')}"
    assert result_json_2.get("d") == {"b": 2, "c": True}, f"Test2 Failed: del animal from d -> {result_json_2.get('d')}"
    assert result_json_2.get("l") == ["long cat", 2, True], f"Test2 Failed: list check -> {result_json_2.get('l')}"
    assert result_json_2.get("__STATEMENT__") == "The long cat plays 2 ball(s).", f"Test2 Failed: statement -> {result_json_2.get('__STATEMENT__')}"
    print("✅ Test Case 2: PASS")


    # --------------------------------------------------
    # テストケース 3: Amend 文 (文修正 & 変数内修正)
    # --------------------------------------------------
    print("\n--- [Test Case 3] Amend Statement Test ---")
    
    scenario_3 = """The long cat plays a ball.
[CODE]amend a ball -> 2 balls"""

    dummy_model_3 = DummyModel(scenario_3)
    messages_3 = [
        {"role": "user", "content": [{"type": "text", "text": "Run amend test."}]}
    ]
    result_json_3, dsl_prog_3 = do_gemma_4_line_by_line_inference(messages_3, dummy_tokenizer, dummy_model_3)
    assert result_json_3.get("__STATEMENT__") == "The long cat plays 2 balls.", f"Test3-1 Failed: {result_json_3.get('__STATEMENT__')}"

    # 変数の amend テスト
    scenario_3_2 = """[CODE]$balls = str two balls
[CODE]amend $balls: two -> three
The long cat plays $balls."""
    
    dummy_model_3_2 = DummyModel(scenario_3_2)
    messages_3_2 = [
        {"role": "user", "content": [{"type": "text", "text": "Run variable amend test."}]}
    ]
    result_json_3_2, dsl_prog_3_2 = do_gemma_4_line_by_line_inference(messages_3_2, dummy_tokenizer, dummy_model_3_2)
    assert result_json_3_2.get("balls") == "three balls", "Test3-2 Failed: var amend"
    assert result_json_3_2.get("__STATEMENT__") == "The long cat plays three balls.", "Test3-2 Failed: statement"
    print("✅ Test Case 3: PASS")


    # --------------------------------------------------
    # テストケース 4: STATEMENT ブロック & return
    # --------------------------------------------------
    print("\n--- [Test Case 4] STATEMENT block & Return ---")
    
    scenario_4 = """[STATEMENT]The long cat plays $balls ball(s).
[CODE]amend $balls -> 2
[STATEMENT]The dog plays with the cat.
[CODE]return 99"""

    dummy_model_4 = DummyModel(scenario_4)
    messages_4 = [
        {"role": "user", "content": [{"type": "text", "text": "Statement and return."}]}
    ]
    result_json_4, dsl_prog_4 = do_gemma_4_line_by_line_inference(messages_4, dummy_tokenizer, dummy_model_4)
    assert result_json_4.get("__STATEMENT__") == "The long cat plays 2 ball(s). The dog plays with the cat.", f"Test4 Failed: {result_json_4.get('__STATEMENT__')}"
    assert result_json_4.get("__RETURN__") == 99, "Test4 Failed: return"
    print("✅ Test Case 4: PASS")


    # --------------------------------------------------
    # テストケース 5: ロボット (一時的なエラーの処理成功ケース)
    # --------------------------------------------------
    print("\n--- [Test Case 5] Robot Exception Recovery ---")
    
    scenario_5 = """[CODE]import tensor
[CODE]import robot1
[CODE]t1 = tensor robot1_normal_act()
[COMMENT]if robot1 should choose walking recovery then use robot1_walk_error_recovery.
[CODE]print robot1_error_router_determine()
[RESULT]"walking"
[CODE]loss = tensor robot1_walk_error_recovery()
[CODE]robot1_error_router_reinforce($loss)
[RESULT]None"""

    dummy_model_5 = DummyModel(scenario_5)
    messages_5 = [
        {"role": "user", "content": [{"type": "text", "text": "Robot action sequence."}]}
    ]
    result_json_5, dsl_prog_5 = do_gemma_4_line_by_line_inference(messages_5, dummy_tokenizer, dummy_model_5)
    
    assert "__ERROR__" not in result_json_5, f"Test5 Failed: __ERROR__ should be resolved and cleared. (Found: {result_json_5.get('__ERROR__')})"
    assert result_json_5.get("loss") == "robot1_tensor_walk_recovery_completed", "Test5 Failed: Robot function call"
    assert "[RESULT]None" in dsl_prog_5, "Test5 Failed: No RESULT"
    print("✅ Test Case 5: PASS")


    # --------------------------------------------------
    # テストケース 6: 回復不能なエラー (多様な raise 表記のテスト)
    # --------------------------------------------------
    print("\n--- [Test Case 6] Unhandlable Error ---")
    
    scenarios = [
        # パターン1: raise Error: Unknwon Error.
        '''[CODE]import tensor
[CODE]import robot1
[CODE]print robot1_unhandlable_error()
[RESULT]"robot1_unhandlable_error_detected"
[EXCEPTION]error: Unknown Error.
[CODE]raise Error: Unknwon Error.''',
        
        # パターン2: raise "Error: Unknwon Error."
        '''[CODE]import tensor
[CODE]import robot1
[CODE]print robot1_unhandlable_error()
[RESULT]"robot1_unhandlable_error_detected"
[EXCEPTION]error: Unknown Error.
[CODE]raise "Error: Unknwon Error."''',
        
        # パターン3: raise Error: "Unknwon Error."
        '''[CODE]import tensor
[CODE]import robot1
[CODE]print robot1_unhandlable_error()
[RESULT]"robot1_unhandlable_error_detected"
[EXCEPTION]error: Unknown Error.
[CODE]raise Error: "Unknwon Error."''',
    ]

    for idx, sc in enumerate(scenarios, 1):
        print(f"  -> Sub-case {idx} test...")
        dummy_model_6 = DummyModel(sc)
        messages_6 = [
            {"role": "user", "content": [{"type": "text", "text": "Unhandlable robot exception."}]}
        ]
        result_json_6, dsl_prog_6 = do_gemma_4_line_by_line_inference(messages_6, dummy_tokenizer, dummy_model_6)
        assert result_json_6.get("__ERROR__") == "Error: Unknwon Error.", f"Sub-case {idx} Failed: Expected 'Error: Unknwon Error.', but got {result_json_6.get('__ERROR__')}"
    
    print("✅ Test Case 6: PASS")


    # --------------------------------------------------
    # テストケース 7: プレーンテキストの改行連続結合 & RESULT改行エスケープ
    # --------------------------------------------------
    print("\n--- [Test Case 7] Text Newline Connection & RESULT Escape Check ---")
    
    scenario_7 = """[CODE]msg = str "line1\\nline2"
[CODE]print msg
[RESULT]line1\\nline2
This is statement 1.
This is statement 2."""

    dummy_model_7 = DummyModel(scenario_7)
    messages_7 = [
        {"role": "user", "content": [{"type": "text", "text": "Verify newlines in DSL."}]}
    ]
    result_json_7, dsl_prog_7 = do_gemma_4_line_by_line_inference(messages_7, dummy_tokenizer, dummy_model_7)
    
    print("\n--- [Generated DSL Program] ---")
    print(dsl_prog_7)
    print("-------------------------------")
    
    assert result_json_7.get("msg") == "line1\nline2", f"Test7 Failed: msg assignment -> {repr(result_json_7.get('msg'))}"
    assert result_json_7.get("__STATEMENT__") == "This is statement 1.\nThis is statement 2.", f"Test7 Failed: statement newline -> {repr(result_json_7.get('__STATEMENT__'))}"
    print("✅ Test Case 7: PASS")


    # --------------------------------------------------
    # テストケース 8: ドット表記による多階層プロパティ・インデックス・動的インデックスアクセス & ネストdel
    # --------------------------------------------------
    print("\n--- [Test Case 8] Dot Notation Access & Nested del Specification Check ---")
    
    scenario_8 = """[CODE]d = dict animal:cat, user:dict name:Alice age:20
[CODE]l = list long cat, 2, true
[CODE]print $d.animal
[RESULT]cat
[CODE]print $d.user.name
[RESULT]Alice
[CODE]print $l.0
[RESULT]long cat
[CODE]print $l.(1+1)
[RESULT]true
[CODE]del d.user.age
[CODE]del l.1"""

    dummy_model_8 = DummyModel(scenario_8)
    messages_8 = [
        {"role": "user", "content": [{"type": "text", "text": "Run Dot Chain and Nested del check."}]}
    ]
    result_json_8, dsl_prog_8 = do_gemma_4_line_by_line_inference(messages_8, dummy_tokenizer, dummy_model_8)
    
    print("\n--- [Generated DSL Program] ---")
    print(dsl_prog_8)
    print("-------------------------------")
    
    assert result_json_8.get("d") == {"animal": "cat", "user": {"name": "Alice"}}, f"Test8 Failed: nested dict check -> {repr(result_json_8.get('d'))}"
    assert result_json_8.get("l") == ["long cat", True], f"Test8 Failed: nested list check -> {repr(result_json_8.get('l'))}"
    print("✅ Test Case 8: PASS")


    # --------------------------------------------------
    # テストケース 9: APP-16 多重にネストされた変数の参照と評価（5つのバリエーション）
    # --------------------------------------------------
    print("\n--- [Test Case 9] APP-16 Multi-nested Variable Reference & Evaluation (5 Variations) ---")
    
    scenario_9 = """[CODE]d = dict name:Alice age:20
[CODE]key = str name
[CODE]print $d.($key)
[RESULT]Alice
[CODE]l = list list a, b, list c, d
[CODE]idx1 = int 1
[CODE]idx2 = int 0
[CODE]print $l.($idx1).($idx2)
[RESULT]c
[CODE]config = dict dev:dict host:dev.local, prod:dict host:prod.live
[CODE]env = str prod
[CODE]print $config.($env).host
[RESULT]prod.live
[CODE]theme = dict light:white, dark:black
[CODE]current_mode = str dark
[CODE]print $theme.($current_mode)
[RESULT]black
[CODE]lang = dict en:Hello, jp:こんにちは
[CODE]user_lang = str jp
[CODE]print $lang.($user_lang)
[RESULT]こんにちは"""

    dummy_model_9 = DummyModel(scenario_9)
    messages_9 = [
        {"role": "user", "content": [{"type": "text", "text": "Run APP-16 nested evaluation test."}]}
    ]
    result_json_9, dsl_prog_9 = do_gemma_4_line_by_line_inference(messages_9, dummy_tokenizer, dummy_model_9, max_lines=25)
    
    print("\n--- [Generated DSL Program] ---")
    print(dsl_prog_9)
    print("-------------------------------")
    
    assert result_json_9.get("d") == {"name": "Alice", "age": 20}, f"Test9 Failed: d -> {repr(result_json_9.get('d'))}"
    assert result_json_9.get("key") == "name", f"Test9 Failed: key -> {repr(result_json_9.get('key'))}"
    assert result_json_9.get("l") == [["a", "b"], ["c", "d"]], f"Test9 Failed: l -> {repr(result_json_9.get('l'))}"
    assert result_json_9.get("config") == {"dev": {"host": "dev.local"}, "prod": {"host": "prod.live"}}, f"Test9 Failed: config -> {repr(result_json_9.get('config'))}"
    assert result_json_9.get("env") == "prod", f"Test9 Failed: env -> {repr(result_json_9.get('env'))}"
    assert result_json_9.get("theme") == {"light": "white", "dark": "black"}, f"Test9 Failed: theme -> {repr(result_json_9.get('theme'))}"
    assert result_json_9.get("current_mode") == "dark", f"Test9 Failed: current_mode -> {repr(result_json_9.get('current_mode'))}"
    assert result_json_9.get("lang") == {"en": "Hello", "jp": "こんにちは"}, f"Test9 Failed: lang -> {repr(result_json_9.get('lang'))}"
    assert result_json_9.get("user_lang") == "jp", f"Test9 Failed: user_lang -> {repr(result_json_9.get('user_lang'))}"
    print("✅ Test Case 9: PASS")


    # --------------------------------------------------
    # テストケース 10: タイポ・未定義識別子による NameError 検証
    # --------------------------------------------------
    print("\n--- [Test Case 10] Typo / Undefined Identifier NameError Verification ---")
    
    scenario_10 = """[CODE]x = int 10
[CODE]print pritn"""

    dummy_model_10 = DummyModel(scenario_10)
    messages_10 = [
        {"role": "user", "content": [{"type": "text", "text": "Run Typo check."}]}
    ]
    result_json_10, dsl_prog_10 = do_gemma_4_line_by_line_inference(messages_10, dummy_tokenizer, dummy_model_10)
    
    assert "__ERROR__" in result_json_10, f"Test10 Failed: Error must be raised -> {repr(result_json_10)}"
    assert "Name 'pritn' is not defined" in result_json_10.get("__ERROR__"), f"Test10 Failed: Expected NameError, got -> {repr(result_json_10.get('__ERROR__'))}"
    print("✅ Test Case 10: PASS")


    # --------------------------------------------------
    # テストケース 11: raise 後は後続 DSL が実行されないことを確認
    # --------------------------------------------------
    print("\n--- [Test Case 11] Abort Stops Subsequent Execution ---")

    scenario_11 = """[CODE]x = int 1
[CODE]raise Error: Stop.
[CODE]x = int 999
This statement should never be added.
[CODE]y = int 2"""

    dummy_model_11 = DummyModel(scenario_11)
    messages_11 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Verify abort behavior."
                }
            ]
        }
    ]

    result_json_11, dsl_prog_11 = do_gemma_4_line_by_line_inference(
        messages_11,
        dummy_tokenizer,
        dummy_model_11
    )

    print("\n--- [Generated DSL Program] ---")
    print(dsl_prog_11)
    print("-------------------------------")

    assert result_json_11.get("x") == 1, \
        f"Test11 Failed: x should remain 1 -> {repr(result_json_11.get('x'))}"

    assert "y" not in result_json_11, \
        "Test11 Failed: y must not be assigned after raise."

    assert result_json_11.get("__ERROR__") == "Error: Stop.", \
        f"Test11 Failed: __ERROR__ -> {repr(result_json_11.get('__ERROR__'))}"

    assert result_json_11.get("__STATEMENT__") == "", \
        f"Test11 Failed: Statement must remain empty -> {repr(result_json_11.get('__STATEMENT__'))}"

    print("✅ Test Case 11: PASS")

    print("\n🎉 すべてのテストケースをクリアしました！例外と回復処理 of ライフサイクル管理は完璧です。")


if __name__ == "__main__":
    run_tests()
