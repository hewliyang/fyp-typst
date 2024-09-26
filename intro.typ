#set heading(numbering: "1.")

#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge
#import fletcher.shapes: diamond

= Introduction
#lorem(120)

== Text to Speech
#lorem(120)

== Audio Signals
#lorem(120)

== Representations for Deep Learning
#lorem(120)

== Motivations and Background
#lorem(120)

+ *Banana*
  - Apple
  - Pineapple

+ Testing 123

#grid(
  columns: (1.5fr, 2fr),
  column-gutter: 10pt,
  [#figure(
      caption: [An Example Table],
      kind: table,
      [
        #table(
          columns: 3,
          [Column A],
          [Column B],
          [Column C],
          table.hline(start: 0, stroke: 1pt),
          [Hi],
          [There],
          [Can],
          [You],
          [See],
          [This],
          [Table],
          [Or],
          [Not?],
        )
      ],
    )<tab-exampleTable>],
  [#figure(
      caption: [A Second Table],
      kind: table,
      [
        #table(
      columns: 3,
      [*Column A*],
      [Column B],
      [Column C],
      // table.hline(start: 0, stroke: 1pt),
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
      [#lorem(2)],
    )
      ],
    )<tab-secondTable>],
)

The results are shown in @tab-exampleTable.

#figure(
  caption: [A graph layout],
  kind: "diagram",
  supplement: [Figure],
  [#diagram(
      node-stroke: .1em,
      node-fill: gradient.radial(blue.lighten(80%), blue, center: (30%, 20%), radius: 80%),
      spacing: 4em,
      edge((-1, 0), "r", "-|>", `open(path)`, label-pos: 0, label-side: center),
      node((0, 0), `reading`, radius: 2em),
      edge(`read()`, "-|>"),
      node((1, 0), `eof`, radius: 2em),
      edge(`close()`, "-|>"),
      node((2, 0), `closed`, radius: 2em, extrude: (-2.5, 0)),
      edge((0, 0), (0, 0), `read()`, "--|>", bend: 130deg),
      edge((0, 0), (2, 0), `close()`, "-|>", bend: -40deg),
    )],
)<fig-banana>

The results are shown in @fig-banana.

#lorem(30)

Here is an in-line equation: $f(x)=sin(x)$ #lorem(10) $g(x)=cos(x)$. #lorem(20).

And here is an out of line equation

$
  f(x) = sin(x)
$<eq-sin>

$
  g(x) = e^(3x pi)
$<eq-exponent>

$
  h(x) = integral_0^1 tan(x) upright(d)x
$<eq-integral>

$
  delta = sqrt(x^2 + y^2)
$<eq-distance>

Refer to @eq-distance.

The first paper is @vaswani2023attentionneed