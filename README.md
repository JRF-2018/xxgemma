# xxGemma

<!-- Time-stamp: "2026-07-08T19:04:26Z" -->

**現在、建設中！**

xxLLM (JSON を出力する xLLM に対し、JSON を出力するプログラムを出力する LLM) を Gemma のファインチューニングで実装する。そのプロジェクトを xxGemma と呼ぶ。このデータセットはそのファインチューニング用のデータセットである。

このデータセットは私が定義した DSL 仕様をもとに、Gemini 3.5 Flashさんに列挙してもらったシチュエーションを ChatGPT 5.5(?) さんに DSL プログラムにしてもらったもの。

例えば、次のような DSL プログラムを出力するように訓練する。

```
[CODE]$animal = str "long cat"
The $animal
[CODE]show 1 + 1
[RESULT]2
[CODE]$balls = int $RESULT
plays $balls balls.
```

…はインタプリタによって、`{"animal":"long cat", "balls": 2, "__STATEMENT__": "The long cat plays 2 balls."}` という JSON を出力する。

また、これだけだと心許ないのでロボット関連に使えないか…と考えて次のような DSL プログラムも理解するように訓練した。

```
[CODE]import tensor
[CODE]import robot1
[CODE]$t1 = tensor robot1_normal_act()
[EXCEPTION]robot1_exception: Robot stumbled.
[COMMENT]if robot1 should choose walking recovery then use robot1_walk_error_recovery.
[CODE]show robot1_error_router_determine()
[RESULT]"walking"
[CODE]$loss = tensor robot1_walk_error_recovery()
[CODE]robot1_error_router_reinforce($loss)
[RESULT]None
```

この場合、出力する JSON は重要でなく、DSL プログラムがある種のルーターとして機能することを想定している。

ちなみに COMMENT や「ヘッダ」にある変数定義部はある種 CoT のように機能することを期待している。この発想には、私が昔、定義部分だけ書いてプログラムを書いた気分になっていたことを思い出す。


## ファイル

`GENERATE_EXAMPLES_2.md` を読めば、一番、全体の状況が見渡せるだろう。

  * `xxgemma_sft_dataset.json`: 今回の主な生成物。Gemma を SFT するためのデータセット。

  * `xxgemma_sft.ipynb`: Colab で GPU L4 を使って SFT する IPYNB。簡単な例も動かす。

  * `xxgemma_interpreter.py`: Gemini さんに作ってもらった xxgemma のインタプリタ。本プロジェクトの核とも言える。`xxgemma_sft.ipynb` にもそのまま含まれている。
  
  * `INTERPRETER.md`: Gemini さんに作ってもらうために書いた DSL 仕様の載ったインタプリタの定義。

  * `GENERATE_EXAMPLES_1.md`: Gemini さんにデータセット用のシチュエーションを定義してもらうために作った状況説明書。

  * `xxGemma_SFT_situations.md`: Gemini さんが出力したシチュエーション集。

  * `GENERATE_EXAMPLES_2.md`: ChatGPT さんにデータセットを具体的に作ってもらうために、DSL 仕様と共にシチュエーション集を載せた状況説明書。
  
  * `README.md`: このファイル。


## Author

JRF ( http://jrf.cocolog-nifty.com/statuses , Twitter (X): @jion_rockford )


## License

私自身は Public Domain にしたいのですが、それだと逆に扱いにくいという場合、AI 作成に問題を関じる場合などは、MIT License でお願いします。

ChatGPT 5.5 さん、Gemini 3.5 Flash さん、Claude Sonnet 4.6 さんにお願いして作りました。


