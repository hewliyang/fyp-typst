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

#include "front-page.typ"

#pagebreak()

#outline(title: "Table of Contents", depth: 3, indent: true)

#pagebreak()

#include "summary.typ"

#pagebreak()

#include "abstract.typ"

#pagebreak()

#set page(numbering: "1", header: align(right)[Evaluating Synthetic Speech])

#include "intro.typ"

#pagebreak()

#include "preliminaries.typ"

#pagebreak()

#include "related-work.typ"

#pagebreak()
#include "dataset-curation.typ"

#pagebreak()

#include "ablations.typ"

#pagebreak()

#include "conclusion.typ"
#pagebreak()

#bibliography("references.bib", style: "springer-basic-author-date")

#pagebreak()

#outline(title: "Tables", target: figure.where(kind: table))

#pagebreak()

#outline(title: "Images", target: figure.where(kind: image))

#pagebreak()

#outline(title: "Diagrams", target: figure.where(kind: "diagram"))
