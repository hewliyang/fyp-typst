#set par(justify: true, linebreaks: auto)
#set text(font: "New Computer Modern", size: 12pt)
#set page(paper: "a4", margin: (x: 1in, y: 1in))
#set figure(gap: 12pt)
#set math.equation(numbering: "(1)", supplement: [Equation])
#show heading: it => {
  it
  v(1em)
}
#show link: box.with(stroke: 1pt + blue, outset: (bottom: 1.5pt, x: .5pt, y: .5pt))
#show cite: box.with(stroke: 1pt + green, outset: (bottom: 1.5pt, x: .5pt, y: .5pt))
#show ref: box.with(stroke: 1pt + red, outset: (bottom: 1.5pt, x: .5pt, y: .5pt))
#let newPage = pagebreak

#include "front-page.typ"

#newPage()

#outline(title: "Table of Contents", depth: 999, indent: true)

#newPage()

#include "abstract.typ"

#newPage()

#set page(numbering: "1", header: align(right)[Evaluating Synthetic Speech])

#include "intro.typ"

#newPage()

#include "preliminaries.typ"

#newPage()

#include "related-work.typ"

#newPage()
#include "dataset-curation.typ"

#newPage()

#include "ablations.typ"

#newPage()

#include "self-supervised.typ"

#newPage()

#include "conclusion.typ"
#newPage()

#bibliography("references.bib", style: "springer-basic-author-date")

#newPage()

#outline(title: "Tables", target: figure.where(kind: table))

#newPage()

#outline(title: "Images", target: figure.where(kind: image))