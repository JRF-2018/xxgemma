# xxGemma

<!-- Time-stamp: "2026-07-10T23:10:43Z" -->

xxLLM (JSON を出力する xLLM に対し、JSON を出力するプログラムを出力する LLM) を Gemma のファインチューニングで実装する。そのプロジェクトを xxGemma と呼ぶ。このデータセットはそのファインチューニング用のデータセットである。

ただし、今回の実装は、相対的に能力の低い小さい Gemma による実装で、実用的な実装とは言えず、PoC (Proof of Concept) 的な実装に留まっている。もちろん、将来的には小さく速い LLM で xxLLM が十分に機能してほしいというのはあるのだが。

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

ちなみに COMMENT や「ヘッダ」にある変数定義部はある種 CoT のように機能することを期待している。この発想は、私が昔、定義部分だけ書いてプログラムを書いた気分になっていたことを思い出させる。


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


## リンク

xxLLM という発想に最初に私が言及したのは↓。

\[cocolog:94833189](2024年5月)  
《ブレストアイデア。同時応答機械指示付きLLM (題して xLLM)。LLM の入力と出力に、機械指示のための json が付いているイメージ。LLM 付きの AI 家電とかやるには必要なものだと思うのだが。- JRF のひとこと》  
http://jrf.cocolog-nifty.com/statuses/2024/05/post-05a8fc.html
> 拡張同時応答機械指示付きLLM (題して xxLLM)は、一般的なものを学習して、それを各家電・ロボット用にチューニングして使う感じか。

私のブログなどでは、しばしば言及し、例えば…

\[cocolog:95029217](2024年9月)  
《xxLLM(拡張同時応答機械指示付きLLM)について Gemini さんと会話した。度忘れしたトピックを思い出すのを手伝ってもらいながら。出力途中での修正や、子分 AI 出力の利用などを話した。 - JRF のひとこと》  
http://jrf.cocolog-nifty.com/statuses/2024/09/post-a48170.html
> (家電操作などを想定して) JSON を出力する xLLM に対し、JSON を出力するプログラムを出力する LLM 名付けて xxLLM のほうが、性能がいいのではないかと考えたことがあります。なぜなら、LLM は逐次出力するため、最初の出力を間違えないようにすることが難しいのですが、xxLLM だと途中で前の式に関して代入などをするなどして修正できるようになるからです。

また、私の[グローバル共有メモ](http://jrockford.s1010.xrea.com/demo/shared_memo.cgi?cmd=log)では、2025-07-09T04:25:28Z に次のように述べている。

> xxLLM というものを昔考えていた。JSON を出力するプログラムを出力する LLM。ただ、Artifact や Canvas を出力するのはそれにすでに近い。メタ的な要素を持っていると思う。xxLLM というものはだからもうすでにいらなくなっているのかもしれない。それはすでに巨大 LLM の内部にはロジックとしてあるのだろう。

そう半ば諦めつつ(試験的な)実装の機会をうかがっていた。

AI さん達の発展の中、機は熟しつつあり、元々は、xxLLM は GPT2 レベルで試験的に実装してみるつもりでいたのだが、斎藤康毅『[ゼロから作るDeep Learning 6 - LLM編](https://www.amazon.co.jp/dp/4814401612)』を読み、そのデータセットを覗いて、そのデータ量の膨大さに驚き、いったん挫折した。しかし、それを Gemini さんに愚痴ったところ、Gemma 使えばいいんじゃね？…と言われたのが、今回のプロジェクトの発端となった。

今後の xxGemma プロジェクトの更新情報などは↓で。  

\[cocolog:96065263](2026年7月)  
《xxGemma 実験を行った。xxLLM (JSON を出力する xLLM に対し、JSON を出力するプログラムを出力する LLM) を Google の「ローカルLLM」 Gemma のファインチューニングで実装する。そのプロジェクトを xxGemma と呼ぶ。 - JRF のひとこと》  
http://jrf.cocolog-nifty.com/statuses/2026/07/post-ca410c.html


## Author

JRF ( http://jrf.cocolog-nifty.com/statuses , Twitter (X): @jion_rockford )


## License

私自身は Public Domain にしたいのですが、それだと逆に扱いにくいという場合、AI 作成に問題を感じる場合などは、MIT License でお願いします。

ChatGPT 5.5 さん、Gemini 3.5 Flash さん、Gemini 3.1 Pro さん、Claude Sonnet 4.6 & 5 さんにお願いして作りました。

----
(This document is mainly written in Japanese/UTF8.)
