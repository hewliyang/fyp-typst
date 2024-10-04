#set heading(numbering: "1.")

= Dataset Curation

In this section, we aim to reproduce the NISQA train set, including the latest Blizzard Challenges 2020, 2021 and 2023. In addition, each stimulus is tagged with an additional boolean field denoting whether the original authors allow commercial use of the data or is restricted to non-commercial purposes such as academic reasearch only.

The datasets are compiled to a NISQA compliant format for ease of training, and are uploaded to the HuggingFace dataset hub for public access. In particular, a Comma Seperated Value (CSV) file is provided with fields

- `filepath_deg`: the relative file path to each stimulus in the form of a `.wav` file
- `mos`: per-stimuli level MOS-N for the given stimulus
- `permissive`: boolean value, `True` indicating permissive commercial use and `False` otherwise

Each dataset repository contains the necessary information for downloading and extracting the raw data into a training ready format. A summary of the datasets can be found in the following table:

#figure(
  caption: "Summary of Curated Datasets from the NISQA Paper",
  kind: table,
  [
    #set text(size: 10.5pt)
    #table(
      columns: 5,
      [Source], [Years], [HuggingFace Repository], [Size (GB)], [\# Stimuli],
      [Blizzard Challenge],
      [2008, 2009, 2010 #linebreak()2011,2012,2013#linebreak()2016,2019,2020,#linebreak()2021,2023],
      [#link("https://huggingface.co/datasets/hewliyang/nisqa-blizzard-challenge-mos")[hewliyang/nisqa-blizzard-challenge-mos]],
      [2.78GB],
      [13,551],

      [Voice Conversion Challenge (VCC)],
      [2016,2018],
      [#link("https://huggingface.co/datasets/hewliyang/nisqa-vcc-mos")[hewliyang/nisqa-vcc-mos]],
      [2.08GB],
      [20,160],
    )],
)

== Blizzard Challenge

The Blizzard Challenge is an annual competition that focuses on the development and evaluation of TTS systems. It was established in 2005 by the Center for Speech Technology (CSTR) at the University of Edinburgh and continues to attract participants from tertiary institutions and industry labs worldwide. The challenge aims to advance the state of TTS technology, whereby participants present innovative solutions to voice conversion tasks.

The primary goal of the Blizzard Challenge is to promote research in TTS synthesis by providing a standardized dataset and evaluation framework, enabling researchers to compare the performance of various TTS systems on a level playing field. This fosters collaboration, and facilititates the sharing of ideas and techniques.

The structure of the Blizzard Challenge varies year-on-year, but generally includes multiple tracks and subtasks that cover different languages and domains.

It is important to note that the organizers also collect other types of subjective metrics including speaker similarity, intelligibility and objective metrics like WER and CER. For the purposes of this work, these metrics are not included.

The Blizzard Challenge generally includes multiple tracks and subtracks that cover different aspects of TTS, such as low-resource scenarios, multi-lingual and expressive speech subtasks.

Each participants submissions are then curated and a listening test is conducted for evaluation. In the earlier years such as @king2008blizzard, testing was done in a controlled lab environment, but recent Blizzard Challenges such as @perrotin23_blizzard has moved on to crowdsourced evaluation as we have seen previously.

The CSTR \@ The University of Edinburgh hosts records of these tests on their web portal at #link("https://www.cstr.ed.ac.uk/projects/blizzard/data.html").

In general, the files when downloaded and uncompressed has a folder structure like:

```
Blizzard_2008
├── statistics.csv																	// listening test statistics
└── A 																							// system identifier
		└── submission_directory
			└── english 																	// language
				└── full 																		// track name
					└── news 																	// subtask name
						├── news_2008_0046.wav
						├── news_2008_0050.wav									// stimuli, denoted by ID
						└── news_2008_0058.wav
```

Note that the folder structure is actually quite messy and not completely coherant year-on-year and that the illustration above is only for demonstration purposes.

We successfully collected all stimuli from all years, with the exception of 2014 @prahallad2014blizzard and 2015 @prahallad2015blizzard.

This was due to the fact that only system level MOS was available. These years were also the years with the most stimuli, with 8990 and 5200 respectively. For example, in 2014 the challenge had 131 systems and 8990 stimuli, which would mean that for each system, the label for the 8990 distinct stimuli would take on a singular value.

We propose that this is not very meaningful for training as it induces bias through an unbalanced train set, and could be a chief reason for the poor per-stimuli performance of NISQA. On the other hand, we do in fact collect 2016 data despite this, due to the relatively lower number of stimuli.

The following table summarizes the new datasets added to the original Blizzard collection described in @Mittag_2020.

#figure(
  caption: "New datasets added to NISQA Blizzard collection",
  kind: table,
  [
    #set text(size: 10.5pt)
    #table(
      columns: 6,
      [Dataset], [Language], [Total Duration (minutes)], [Avg Duration (s)], [\# Sys], [\# Files],
      [Blizzard 2020], [`zh`], [90.4], [6.8], [26], [795],
      [Blizzard 2021], [`en`, `es`], [48.6], [5.2], [24], [556],
      [Blizzard 2023], [`fr`], [83.4], [3.4], [38], [1460],
    )],
)

Note that the challenge was not conducted in 2022.

== Voice Conversion Challenge (VCC)

The VCC serves a similar purpose to the Blizzard Challenge focusing on subtasks such as cross-lingual voice conversion. In recent years, VCC has pivoted to assesing singing voices @huang2023singingvoiceconversionchallenge and not speech . In addition, the organizers of VCC 2020 @yi2020voice did not release the labeled evaluation sets.

Curating the test sets for VCC was relatively simpler compared to Blizzard. We validate that our metrics match up to the reported numbers in the NISQA paper as well.

#figure(
  caption: "Summary of VCC 2016 and 2018 datasets",
  kind: table,
  [
    #set text(size: 10.5pt)
    #table(
      columns: 6,
      [Dataset], [Language], [Total Duration (minutes)], [Avg Duration (s)], [\# Sys], [\# Files],
      [VCC 2018 HUB], [`en`], [108], [3.46], [52], [2000],
      [VCC 2018 SPO], [`en`], [112], [4.23], [28], [2000],
      [VCC 2016], [`en`], [1428], [3.73], [20], [26028],
    )],
)

Note that for VCC 2016, no per-stimuli level ratings were released. However, we still include it as it was used as a test set and not in training.

== PhySyQx

The PhySyQx project was originally meant to study the relationship between speech quality and brain activitiy, but also happens to include some subjective scores along with it's stimuli, including emotion, valence, comprehension, etc.

We only take the naturalness MOS, of which there are 36 stimuli level ratings which spans about 12 minutes worth of audio data.

