> * 此为技术类笔记，个人对中英文单词互相标注做如下规定：

>    * 文中首次出现，使用**中文术语（english terminology）**形式给出。

>    * 其后使用中，为了方便延伸阅读（尤其是英语论文），以常用形式为标准，部分直接使用英文术语；部分中文术语已经约定俗成并且足够生动形象，如“分词”、“词性标注（part-of-speech tagging）”、“词义消歧（word sense disambiguation）”等，则继续使用中文。

>    * 两种情况下，英文词直接使用，不再给出中文释义：一是如 parsing、target、socket、handle 等短小会意的词；二是在深入某话题后接触到大量的概念 / 前沿新发明的概念等，如 Information Content Similarity，没有翻译的必要。

[TOC]

## 00 课程介绍

* 参考教材
    * Daniel Jurafsky and James H. Martin: Speech and Language Processing, Pearson Prentice Hall; 2 edition, 2008
    * Christopher D. Manning and Hinrich Schuetze: Foundations of Statistical Natural Language Processing. The MIT Press, 1999.

* 高水平学术论文
    * ACL、EMNLP、NAACL、Coling 等。

## 01 计算语义学概述

* **语义学（Semantics）**是对自然语言意义（meaning）的研究。与一般的自然语言处理（NLP）相比偏向于语义理解，因此**计算语义学（Computational Semantic）也可以约等于自然语言理解（Natural Language Understanding，NLU）**。

* 不同领域对语义学的研究侧重不同。语言学、逻辑学、认知科学、计算机科学都有各自的研究。

* 两个人在交谈中，一方是语言生成（language generation），另一方是语义理解（semantic analysis, or understanding），所以本质上是对语义的 encode 和 decode 的过程。

* 这中间的 parsing 过程有点像**编译器**，从 source 到 target 要经历**词法分析（lexical analysis）**、**语法分析（syntax analysis）**、**语义分析（semantic analysis）**等步骤。打个比方，输入`3 + 7 * 8`，**lexical analysis 是分析出一个个 token**，如`3`、`7`、`8`；**syntax analysis 是分析规则**，如先算乘法还是先算加法；**semantic analysis 是分析含义**，如为何`7 * 8`是`56`。尽管比较像编译器，但是计算语义学的工作比编译器要复杂得多，原因在于自然语言比形式语言复杂多变。

* NLP 中有一道道工序，如分词、词性标注（POS tagging）、命名实体识别（NER）、句法分析和语义分析。几乎每道工序都有自己的表示方式，如分词一般用空格隔开，句法分析一般用句法树。只有**语义分析目前没有很明确的表示方式，或者说需要按照不同应用使用不同的表示方式**。

* **计算语义 = 语义计算 = 语义分析 ≈ 语言理解**。另外，词汇语义学（lexical semantic）研究单词的含义，如词义消歧（word sense disambiguation）；组合语义学（compositional semantics）研究单词如何组合以形成更大的语义。

* 语义分析重要性在于，语言是思维的载体，是交流思想、表达情感最直接的工具。人类历史以语言文字为载体的知识占 80% 以上。语义分析的难点在于：歧义、省略、推理、动态变化等。

## 02 基于知识库的词汇语义计算

* 词语（word）有中文词和英文词两类。本课程中以**英文词**为例。

* 英文词形态变化比较多，一个单词可以由**基本形态（lemma）**变化为多个**屈折形态（inflected forms）**，如 get -> gets, got, gotten, getting。因此在处理的时候，有时需要进行**词形还原（lemmatization）**。需要还原的除了一些时态、数量信息，还要删除许多**词根（stem）**如 ed、ing、able、ism 等。注意有些词在 lemmatization 的时候需要考虑是否有必要，如 thought 如果做名词就不需要处理。

* 一个**词语（word）**可能有多个**词义（word sense）**，每个**词义**被一条**释义（gloss）**所描述。**一词多义的情况在英文中细分成 homonym 和 polysemy**，两者都叫多义词，区别是 **homonyms 指两个义项完全不相关**，如 bank 用于 money bank 和 river bank 的情况；**polysemes 指两个义项仍相关**，如 bank 用于 money bank 和 blood bank 的情况。但其实两者之间界限是模糊的。

* 一个 word 有多少 sense 完全是主观的，不同的人、任务下的细粒度会有差异，比如对于单词 drive，WordNet 有达 34 个意义，普通人就不会分这么细。

* 词义之间的基本关系（Semantic relation）有：**同义（syno-nymy）**、**反义（anto-nymy）**、**上位（hyper-nymy）**、**下位（hypo-nymy）**、**整体（holo-nymy）**、**部分（mero-nymy）**。为了方便记忆已经在词根 -nymy [nɪmɪ] 处用“-”隔开了，且将词根 -nymy 替换成 -nym 即是他们的名词，替换成 -nymous 即是他们的形容词。

* 同义反义好理解，虽然按照人来看他们的意思很明确，但如果简单按照代入法来检测（如 I bought/purchased a car.）还是难以区分（如 This is good/bad.）。不存在完美的同义词和反义词；即使是反义词，但在某种角度看他们仍有一定共性（相关性？）。

* 上位下位指的就是父类（如 fruit）和子类（如 apple）。整体和部分则是像 car 和 wheel 的关系。

* 同义词之间的关系可以是二值的（是 / 不是同义关系），也可以用**词汇相似度（word similarity）** / **词汇语义距离（word semantic distance）**计算，注意两者数值是相反的，相似度越高，距离越近/小。计算相似度有两种方法：**基于辞典**（**thesaurus-based**，注意一般翻译成“辞典”而不是“词典”，thesaurus 含有同义词）和**基于统计（distributional/statistical）**。前者是看两个词在词典（如 WordNet）中是否相邻，后者是比较两个词在语料中的上下文。

* [WordNet](http://wordnet.princeton.edu/)
    * **WordNet 的基本单元是 Synset**（synonym set 的合称，即同义词集合），**每个 synset 表示一个语义概念**。
    * **每个词条包含多个 synset（可以视为义项）、gloss（该义项的释义）、sample sentence（该义项的例句）等信息。**
    * synset 和 synset 之间通过多种词义关系（即上述同义反义上位下位整体部分等）相连。
    * **在 WordNet 中只有 4 种词性：Nouns、Verbs、Adjectives、Adverbs。**
    * WordNet 除了英语外，对欧洲语系有 EuroWordNet，亚洲语系只有几种如 Hindi、Marathi、Japanese。
    * WordNet 细粒度非常细。（但是细粒度对 NLP 任务是否有用是不一定的。）

* WordNet-based Word Similarity
    * WordNet 的任何一部分信息都能拿来使用 / 计算，如 relation、gloss、example sentence。
    * 需要区分**相似度（word similarity）**和**相关度（word relatedness）**！
    * Path-based similarity
        * 看两个词在词典中层次结构（如上下位关系树）中的位置，路径越短越相似。
    * Information Content Similarity
        * （待补完）

> **Reference**

>    * Alexander Budanitsky and Graeme Hirst. 2006. Evaluating WordNet-based Measures of Lexical Semantic Relatedness. Comput. Linguist. 32, 1 (March 2006), 13-47. [[pdf]](http://www.mitpressjournals.org/doi/pdfplus/10.1162/coli.2006.32.1.13)

>    * Hughes, Thad, and Daniel Ramage. "Lexical Semantic Relatedness with Random Graph Walks." EMNLP-CoNLL. 2007. [[pdf]](http://www.aclweb.org/anthology/D07-1#page=615)

## 03 词义消歧

> **Reference**

>    * Roberto Navigli. 2009. Word sense disambiguation: A survey. ACM Comput. Surv. 41, 2, Article 10 (February 2009) [[pdf]](https://www.researchgate.net/profile/Roberto_Navigli/publication/220566219_Word_sense_disambiguation_A_survey/links/54bba1370cf253b50e2d1055.pdf)