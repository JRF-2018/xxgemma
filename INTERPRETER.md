# xxGemma プロジェクト: インタプリタの作成

## 目的

xxLLM (JSON を出力する xLLM に対し、JSON を出力するプログラムを出力する LLM) を Gemma のファインチューニングで実装したい。そのプロジェクトを便宜的に今、xxGemma と呼ぶ。

今回は、DSL の仕様からインタプリタを作成していく。

まず DSL の仕様を提示する。次に unsloth で gemma を使って一行ごとに生成するコードを示し、その上にインタプリタを作ってもらう。そして、インタプリタをテストするために、すでにある DSL のコードの会話例が確かに、インタプリタで適切な値が得られるかをテストするために、unsloth の model.generate のダミーモデルを作ってもらう。


## DSL (Domain Specific Language) 仕様

例えば、

The long cat
[CODE]animal = str long cat
plays two balls.
[CODE]balls = int 2

…という文は、

{"animal":"cat", "balls": 2, "__STATEMENT__": "The long cat plays two balls."}

…という JSON になる。変数は $animal などで参照できる。また計算結果の表示もできる。だから次のように書いてもよい。

[CODE]animal = str long cat
The $animal
[CODE]int 1 + 1
[RESULT]2
[CODE]balls = int $RESULT
plays $balls balls.

…という文も、

{"animal":"long cat", "balls": 2, "__STATEMENT__": "The long cat plays 2 balls."}

…という JSON になる。

RESULT は学習時および生成時は、LLM が生成もするが、それは生成時は随時正しい結果に書き換えられるはず。


代入文は最初に $ ではじめてもよい。$ がないものも許容する。ただ、Gemma の SFT 時には $ ではじめるものを使う。

[CODE]$animal = str long cat
The $animal
[CODE]int 1 + 1
[RESULT]2
[CODE]$balls = int $RESULT
plays $balls balls.

…でも許容する。

この後の場合でも $ を付けるか付けないかは柔軟に対応する。学習時は付けれるところでは付けたほうがいいらしい(Geminiさんの見解)。


"" を使いベアワードを避ける方法、関数による生成、\ によるエスケープをサポートしておく。あと変数の中身を見たいときは show も用意しておく。Gemma の SFT 時には、ベアワードではなく "" を使う。

[CODE]animal = str "long cat"
The $animal
[CODE]balls = int 1 + cos(0)
[CODE]show balls
[RESULT]2
plays $RESULT ball\(s\).

…という文は、

{"animal":"cat", "balls": 2, "__STATEMENT__": "The long cat plays 2 ball(s)."}

…という JSON になる。show $balls は許容するし、show 1 + 1 なども許容したほうがいいかもしれない。


デフォルトの「型」は str int float bool list dict がある。あと、dict や現在の json からリムーブするための del 文がある。

[CODE]d = dict $animal, b:$balls
[CODE]l = list $animal, $balls
[CODE]del $balls
[CODE]del animal
[CODE]del animal from d

などと記述できる。

要素へのアクセスは $d.animal $d.b $l.0 $l.(1 + 1) などで。


amend 文で文章を修正できる。

The long cat plays a ball.
[CODE]amend a ball -> 2 balls

…とすると、{"__STATEMENT__": "The long cat plays 2 balls."} が返る。

また $v1 は定義されてないときは $v1 = \$v1 であるかのように振るまい…

The long cat plays $balls ball(s).
[CODE]amend $balls -> 2

…とすると、{"__STATEMENT__": "The long cat plays 2 ball(s)."} が返る。


amend は特定の変数についても修正できる。

[CODE]$balls = str two balls
[CODE]amend $balls: two -> three
The long cat plays $balls.

…とすると、{"__STATEMENT__": "The long cat plays three balls."} が返る。


__STATEMENT__ は途中で $STATEMENT として参照できるようにする。

The long cat plays $balls ball(s).
[CODE]amend $STATEMENT: $balls -> 2

…とすると、やはり {"__STATEMENT__": "The long cat plays 2 ball(s)."} が返る。


STATEMENT ブロックもいちおう用意する。SFT 時には使わないが、インタプリタでは許容する。

[STATEMENT]The long cat plays $balls ball(s).
[CODE]amend $balls -> 2
[STATEMENT]The dog plays with the cat.

…とすると、{"__STATEMENT__": "The long cat plays 2 ball(s). The dog plays with the cat"} が返る。


if 文などの代わりに [COMMENT] 文があったほうがよい。事実上の(定義による以外の) CoT にも使える。

[COMMENT]if $animal is cat, then $balls = 3.
[CODE]show $animal
[RESULT]cat
[CODE]balls = int 3

…みたいに使う。


[CODE]return $d

…もあったほうがいい。

最終的な JSON に __RETURN__ にセットされるだけまたは、その時点で文の生成は終了しても良い。Gemini さんは終了を推すようだ。


インタプリタのエラー終了を表すための JSON の項目 __ERROR__ も必要だろう。


xxLLM ではロボットも扱いたい。そのためテンソルが使える拡張もしたい。そのために import 文を便宜的に用意しておく。これはインタプリタに import を要請するもので、その際 tensor 型の導入も行われる。

[CODE]import tensor
[CODE]import robot1
[CODE]t1 = tensor robot1_scan()
[CODE]show robot1_is_normal(t1)
[RESULT]true
[CODE]t2 = tensor robot1_model(t1)
[CODE]robot1_act(t2)

…などとしたい。


インタプリタの途中でエラーが発生したことなどを LLM に通知するには [EXCEPTION] 文を使う。[EXCEPTION]文はロボット制御などにおいて、途中で処理すべきことが現れた場合の注意などにも使われる。

[CODE]import tensor
[CODE]import robot1
[CODE]$t1 = tensor robot1_normal_act()
[EXCEPTION]robot1_exception: Robot stumbled.
[COMMENT]if robot1 should choose walking recovery then use robot1_walk_error_recovery.
[CODE]show robot1_error_router_determine()
[RESULT]"walking"
[CODE]loss = tensor robot1_walk_error_recovery()
[CODE]robot1_error_router_reinforce($loss)
[RESULT]None

通常のエラーの場合は…

[CODE]$balls = integer 1
[EXCEPTION]error: Syntax Error.
[CODE]$balls = int 1

…など。


[EXCEPTION] について処理しきれない場合、__ERROR__ を JSON に返す。このとき次のようにする。

[CODE]import tensor
[CODE]import robot1
[CODE]show robot1_unhandlable_error()
[EXCEPTION]error: Unknown Error.
[CODE]raise Error: "Unknwon Error".

…とすると

{"__ERROR__": "Error: Unknwon Error."}

…が返る。


自分自身(の前のバージョン)を参照する query 文や、Web を参照する search 文は、時期尚早ということで、実装は見送る。


## インタプリタの作成

一行ごとの生成は次のようなコードでできる。

```
import torch
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList

# 1. 改行コード（\n）が出現したらLLMを一時停止させるための判定クラス
class StopAtNewLineCriteria(StoppingCriteria):
    def __init__(self, tokenizer, start_length):
        self.tokenizer = tokenizer
        self.start_length = start_length

    def __call__(self, input_ids, scores, **kwargs):
        # 今回の generate 呼び出し以降に「新しく生成されたトークン」のみを抽出
        new_tokens = input_ids[0][self.start_length:]
        text = self.tokenizer.decode(new_tokens)

        # 文字列の中に改行コードが含まれたら True を返して生成をストップ
        return "\n" in text

# 1行ずつの生成を管理・インターセプトする関数
def do_gemma_4_line_by_line_inference(messages, max_lines = 10):
    # 初回のプロンプト（チャットテンプレート）を準備
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    )
    input_ids = encoded["input_ids"].to("cuda")

    print("=== 🤖 xxLLM 行ごと生成ループ開始 ===")

    for line_idx in range(max_lines):
        # 現在のコンテキストの長さを記録（ここから先がこのターンで新しく生成される行）
        start_length = input_ids.shape[1]

        # 毎ループ、現在の長さを基準にした停止条件をセット
        stop_criteria = StoppingCriteriaList([StopAtNewLineCriteria(tokenizer, start_length)])

        # 1行分（改行が出るまで）生成
        outputs = model.generate(
            input_ids = input_ids,
            max_new_tokens = 128,
            temperature = 1.0, top_p = 0.95, top_k = 64,
            stopping_criteria = stop_criteria,
            pad_token_id = tokenizer.eos_token_id,
            # skip_prompt=True で、過去の蓄積された文脈が再表示されるのを防ぐ
            streamer = TextStreamer(tokenizer, skip_prompt = True)
        )

        # 生成された内容を含めた全体を、次回の入力文脈として上書き
        input_ids = outputs

        # このターンで新しく生成された1行のテキストを取得
        new_ids = outputs[0][start_length:]
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        # ------------------------------------------------------------------
        # 💡 [ここにPythonインタプリタの処理を挟めます]
        # ------------------------------------------------------------------
        print(f"← 🛠️ [Python制御: {line_idx + 1}行目の生成終了を検知しました]")

        # もし生成された行が DSL のコードを意味していたら……のシミュレーション
        clean_line = new_text.strip()
        if clean_line.startswith("[CODE]"):
            print(f"    ↳ ★DSLコード行を検出: {clean_line}")
            # ここで実行結果を [RESULT] として input_ids の末尾に強制インジェクションするコードを後々挟めます

        # ------------------------------------------------------------------

        # LLMが自発的に終了トークン（EOS）を吐き出していたら、ループを完全に抜ける
        if tokenizer.eos_token_id in new_ids:
            print("=== 🔚 EOS（終了トークン）を検知したため全体を終了します ===")
            break
```

```
messages = [{
    "role": "user",
    "content": [{ "type" : "text", "text" : "Write a 3-line poem about sloths. Line by line." }]
}]
do_gemma_4_line_by_line_inference(messages)
```

出力は次のようになる。

```
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

=== 🤖 xxLLM 行ごと生成ループ開始 ===
Slowly moving through the green,

← 🛠️ [Python制御: 1行目の生成終了を検知しました]
A gentle, sleepy, furry scene,

← 🛠️ [Python制御: 2行目の生成終了を検知しました]
Life in a tranquil, mossy mien.<turn|>
← 🛠️ [Python制御: 3行目の生成終了を検知しました]
=== 🔚 EOS（終了トークン）を検知したため全体を終了します ===

Colab の有料サービス - 契約解除はこちら
```

これを参考にインタプリタを回すことになる。

そのインタプリタを作っていただきたい。

いくつかの初歩的な関数 sin cos などを理解するようにしておいていただきたい。

import すべき robot1 に関しては、今回は、ハードコーディングしておいていただきたい。必要な関数は上に書いたものから類推していただきたい。



## テスト用モデル

model.generate で呼べるようなダミー model を作っていただきたい。応答完成文章を渡して model を作り、model.generate でそれをそのまま作成していくような。

それで例えば

```
messages = [{
    "role": "user",
    "content": [{ "type" : "text", "text" : "Generate text that long cat plays two balls." }]
}, {
    "role": "assistant",
    "content": [{ "type" : "text", "text" : """\
[CODE]animal = str long cat
The $animal
[CODE]int 1 + 1
[RESULT]2
[CODE]balls = int $RESULT
plays $balls balls.
"""}]
}]
```

をインタプリタに食わせて

```
{"animal":"cat", "balls": 2, "__STATEMENT__": "The long cat plays two balls."}
```

が得られるかテストしたい。



