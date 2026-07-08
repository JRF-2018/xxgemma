# xxGemma プロジェクト: SFT 用データセットの作成 1: シチュエーションの列挙

## 目的

xxLLM (JSON を出力する xLLM に対し、JSON を出力するプログラムを出力する LLM) を Gemma のファインチューニングで実装したい。そのプロジェクトを便宜的に今、xxGemma と呼ぶ。

すでに「DSL の仕様からインタプリタを作成する」は終了している。

次に、Gemma の SFT 用に300例のデータセットを作るのを目指す。2ステップで行う。

  1. 例として適当なシチュエーションの列挙。

  2. 一つのシチェーションに基づいた実際の例を json で 5例ほど作っていく。それを全シチュエーションについて繰り返す。

まずこの 1 を行うのが今回の目的である。


シチュエーションは大分類として以下のように分ける。

| カテゴリ | 想定件数 | 構造の特徴 |
| --- | --- | --- |
| **正常系（基本）** | 100件 | 変数代入、数式計算、単純なJSON化 |
| **正常系（応用）** | 80件 | `amend` による文や変数の書き換え、`dict`/`list` 操作 |
| **エラー系（構文）** | 60件 | `Syntax Error` 発生 $\rightarrow$ `[EXCEPTION]` $\rightarrow$ 自分で修正 |
| **エラー系（物理）** | 60件 | `Robot stumbled` 発生 $\rightarrow$ `[COMMENT]`提示 $\rightarrow$ リカバリコード |


これを Canvas でシチュエーション一つを5例用に細かめに作って欲しい。つまり、300/5 = 60 シチュエーションを作って欲しい。


## DSL (Domain Specific Language) 仕様

例えば、

The long cat
[CODE]$animal = str "long cat"
plays two balls.
[CODE]$balls = int 2

…という文は、

{"animal":"cat", "balls": 2, "__STATEMENT__": "The long cat plays two balls."}

…という JSON になる。変数は $animal などで参照できる。また計算結果の表示もできる。だから次のように書いてもよい。

[CODE]$animal = str "long cat"
The $animal
[CODE]show 1 + 1
[RESULT]2
[CODE]$balls = int $RESULT
plays $balls balls.

…という文も、

{"animal":"long cat", "balls": 2, "__STATEMENT__": "The long cat plays 2 balls."}

…という JSON になる。

RESULT は学習時および生成時は、LLM が生成もするが、それは生成時は随時正しい結果に書き換えられるはず。


$ を付けるか付けないかはインタプリタでは柔軟に対応する。ただし SFT 学習時は付けれるところでは付ける。


関数による生成、\ によるエスケープをサポートしておく。あと変数の中身を見たいときは show も用意しておく。"" を使わずベアワードを使う方法もインタプリタでは許容する。ただし、Gemma の SFT 時には、ベアワードは使わず "" を使う。

[CODE]$animal = str "long cat"
The $animal
[CODE]$balls = int 1 + cos(0)
[CODE]show $balls
[RESULT]2
plays $RESULT ball\(s\).

…という文は、

{"animal":"cat", "balls": 2, "__STATEMENT__": "The long cat plays 2 ball(s)."}

…という JSON になる。


デフォルトの「型」は str int float bool list dict がある。あと、dict や現在の json からリムーブするための del 文がある。

[CODE]$d = dict $animal, b:$balls
[CODE]$l = list $animal, $balls
[CODE]del $balls
[CODE]del animal
[CODE]del animal from $d

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


if 文などの代わりに [COMMENT] 文があるといえる。事実上の(定義による以外の) CoT にも使える。

[COMMENT]if $animal is cat, then $balls = 3.
[CODE]show $animal
[RESULT]cat
[CODE]$balls = int 3

…みたいに使う。


[CODE]return $d

…もあったほうがいい。

最終的な JSON に __RETURN__ にセットされるだけまたは、その時点で文の生成は終了しても良い。Gemini さんは終了を推すようだ。


インタプリタのエラー終了を表すための JSON の項目 __ERROR__ も必要だろう。


xxLLM ではロボットも扱いたい。そのためテンソルが使える拡張もしたい。そのために import 文を便宜的に用意しておく。これはインタプリタに import を要請するもので、その際 tensor 型の導入も行われる。

[CODE]import tensor
[CODE]import robot1
[CODE]$t1 = tensor robot1_scan()
[CODE]show robot1_is_normal($t1)
[RESULT]true
[CODE]$t2 = tensor robot1_model($t1)
[CODE]robot1_act($t2)

…などとしたい。


インタプリタの途中でエラーが発生したことなどを LLM に通知するには [EXCEPTION] 文を使う。[EXCEPTION]文はロボット制御などにおいて、途中で処理すべきことが現れた場合の注意などにも使われる。

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


## Tips

jrf:＞＞記憶というか学習素子というかは、覚えやすいもの、覚えにくいものがあるように思います。それ自身が学習されている感じを持ちます。

最適化…ってヘシアンまで見るじゃないですか？ 機械学習においても、そんな風に覚えにくさというものを各行列のパラメータごとに持っている構造って、すでに考えられてるように思うのですが、どうでしょう？ それの研究の進み具合みたいなのを教えてください。
＜＜

Gemini:＞＞Adam / RMSprop、Shampoo / K-FAC、EWC（Elastic Weight Consolidation） / Synaptic Intelligence などがあります。

JRFさんの amend 構文は、低層のパラメータに個体差を持たせる代わりに、「文法（DSLというソフトウェアの幾何学）によって、モデルに『ここは後で埋める穴（未定義変数）だから、今は適当にスルーして（覚えずに）先に進め』という記憶の制御を外付けで行っている」と解釈することができます。

重いヘシアンの計算を回さずとも、記号の構造だけでモデルの「待機と発出」をコントロールする。こうして既存の研究の地平と比較してみても、JRFさんのアプローチは非常にエコノミカルで、かつ批評的な立ち位置にあることが分かります。

ニューラルネットワークの重み空間（パラメータ空間）は、数兆次元の「滑らかな地形で構成された連続的な空間」です。

ここで「ある知識を覚えやすく（あるいは変わりづらく）する」ということは、その多次元空間の特定の谷底の「斜面の傾きや曲率（ヘシアン）」を計算し、「この方向にはパラメータを動かすな」という数理的な粘性（バネの硬さ）を重み一つひとつに直接付与することを意味します。これは、アトム（物理）のレイヤーでセメントを流し込んで固めるような力技です。

一方、amend 構文が操作しているのは、パラメータの連続空間ではなく、その上に浮遊する「離散的なトークン（記号）の並び」です。

文章の中に $balls という未定義変数をポツンと置く行為は、数理的には「確定していない自由度（穴）を、コンテキスト（文脈）の中に意図的に残したまま、次のトークン予測へと確率の波を進める」という処理になります。

重み自体を固めるのではなく、「文法（記号の並びルール）」という外付けのレールによって、モデルの視線（Attention）の進路を強制誘導しているわけです。
＜＜

jrf:＞＞Gemini さんは amend に強い関心を持っておられるようですが、その前のヘッダ部分による定義というのも結構重要なように思います。ここは言ってしまえば CoT なんですね。

昔、定義部分だけ書いてプログラムを書いた気分になっていたのを思い出します。
＜＜

……。

普通のニューラルネットをこの枠組みにそのまんま持ってくることを考えると、RESULT で途中結果を数値で示すことなしに、入力を NN の式でプログラム的に変形して出力まで変換する。その NN の式が xxLLM の生成内容であり、そのパラメータはすでにある。…つまり、NN のパラメータを学習するのはプログラム変換の責任という形が素直な形となるでしょう。

xxLLM がやるべきことはせいぜい、指示からどの LoRA を選ぶかの補助をする…ぐらいになるのかもしれません。

プログラム変換は、勾配降下法などで普通に学習していく…。xxLLM が覚えるのは NN のトポロジー程度…。しかし、両者共に学習できることを考えれば、これまでとまったく違うトポロジーも可能になりそうなものですが…。絵をただ記憶するしかないと思われていた AI が、実際には創造性を持った。そんなことがトポロジーについても起きないものでしょうか…。

……。

プログラム変換をするプログラム(AI)は、いくつもの NN を知り、それを使い分けられたほうがいいでしょう。できれば、さらにいくつかの NN の目的に沿った合成ができればベターです。まぁ、合成といってもどの LoRA を使うかぐらいの話に今の時点ではなるのかもしれませんが。

プログラム変換のプログラム(AI)が通常の学習・生成より有利な点は、データセット全体、ツールセット全体を見渡して使ってもいいこと、スピードをあまり求められないことでしょうね。

要するに、xxLLM はそういうプログラム変換プログラム(AI)のある種の蒸留だということになるのでしょう。数ある蒸留の中で特に優秀または効率的な蒸留になりうるかが、成功か否かの目安になるのかもしれません。

ChatGPT:＞xxLLM の価値は「計算を代行すること」よりも、「構造を保存し、再利用し、蒸留すること」にあるのかもしれません。そこまで行くと、DSLというより「学習された構造の記述言語」という位置づけに近づいてきます。＜

……。

おそらくロボットも MoE 的なルーターが大事なんだろうな。介護で人を持ち上げるとき、手の動きのモデルを先に決めて、それを元に安定させるような中腰の下半身のモデルを先の手の動きの出力と共に使う…みたいなのを、動作ごとに切り替えるような。指の動きのモデル、手の動きのモデル、足の支えのモデルの順に送っていくが相互作用もある…みたいなのもルーティングする必要があるのだろう。そして、細かい部分では RLRMDiffusion (RL Result Model Diffusion) みたいに LoRA を生成したりしながら対処もして、どういう風に生成すればよいかもルーティングするのだろう。

……。

ロボットのルーターで MoE 的に素朴な物は、歩いてる途中につまづいたときに、その「エラー」処理を、歩く処理のモデルにエラーからの回復をまかせるか、バランスを回復する処理のモデルに歩くのも若干加味させるか…といったところから選択する…といったものが考えられる。

こういったものから先の介護での利用になると、以前述べた xxLLM を使ったようなルーターが必要になるのだろう。
