# Writing checklist — van Gemert, *Research Guidelines in Deep Learning*

Condensed from https://jvgemert.github.io/ResearchGuidelinesInDL.pdf (read in full, 2026-06-10).
Every section of the thesis is checked against this file before being shown for review.
Codes match the PDF so feedback can point at items directly.

## Storyline (§2 of the PDF — the spine of the paper)

The storyline is a minimal, stand-alone, fully-motivated narrative; no jargon term used before
it is motivated/introduced; every claim can be challenged, so every claim is motivated or cut
(Hitchens's razor: asserted without evidence ⇒ dismissed without evidence).

- [ ] S1 **Why interesting?** Tightly-scoped motivation; open the abstraction to the concrete
      core ("much research on X" is not a reason). Scope claims must come back in experiments.
- [ ] S2 **How done now?** Relevant (not exhaustive) current approaches, described objectively,
      factually (their authors would agree).
- [ ] S3 **What is missing, and So What?** The problem with S2 + its precise consequences,
      linked back to S1; consequences are the concrete things experiments will show.
- [ ] S4 **Proposed approach.** Why it addresses S3 (the "why", not the "what"); a logically
      consistent "house of whys" understandable by non-experts.
- [ ] S5 **Experimental questions.** Controlled: validate the problem occurs and the approach
      addresses it. Uncontrolled: does it hold in 'real' settings?
- Content problems to avoid: unclear why-interesting; vague how-done-now; problem/consequences
  not linked to motivation; unclear proposal; unclear why-it-solves; unclear experimental
  questions; irrelevant storyline parts.
- Form problems to avoid: term before definition; unnecessary detail; unclear reasoning steps;
  inconsistent terminology (one term per concept, 1-to-1).

## Research process & mentality (selected, paper-relevant)

- [ ] RP6 Each experiment answers a single, explicitly-written question (expected answer
      written down beforehand).
- [ ] RP9 Change only one variable, else the cause of an effect is indeterminable.
- [ ] RP11 Script ALL figures (matplotlib → PDF), never hand-make.
- [ ] RM1 Re-read as a savage reviewer looking for any excuse to reject.
- [ ] RM3 Be consistent: assumptions do not silently change between sections.
- [ ] RM5 Simple is strong; explainable to a smart layperson without jargon.
- [ ] RM6 Identify limitations explicitly; showing where it fails is strength.
- [ ] RM10 Motivate everything: each choice has a reason; commonly done ⇒ cite it;
      speculative ⇒ validate it as a hypothesis or label it.

## General writing (WG)

- [ ] WG1 Unburden the reader: misinterpretation is the writer's fault; reader does no work.
- [ ] WG2 Know the audience (CS-bachelor examiner; non-DL-expert must follow intro/conclusion).
- [ ] WG3 Less is more: every word has a reason to exist.
- [ ] WG4 No guessing: explicitly write what the reader should see/conclude.
- [ ] WG5 Read out loud before delivering.
- [ ] WG6 Important topics take more space; unimportant ones less.
- [ ] WG7 Writing is like coding: restructure/refactor text in passes.

## Structure (WS)

- [ ] WS1 Self-contained: remind of definitions/symbols defined long ago.
- [ ] WS2 Use a defined symbol consistently and uniquely.
- [ ] WS3 Avoid "as discussed before / as will be described in §X" — the standard section
      structure already dictates where information lives; paper is not read linearly.
- [ ] WS4 Paragraph topics follow a logical order (validate with an inverse outline).
- [ ] WS5 Avoid reference words (this/it/that/there) — repeat what is meant explicitly.
- [ ] WS6 No reference words across paragraphs (never start a paragraph with "However").
- [ ] WS7 Avoid "latter/former" — no mental ordering work for the reader.

## Form (WF)

- [ ] WF1 One paragraph = one topic: intro sentence defines topic, each sentence follows the
      previous, concluding sentence answers "So what?".
- [ ] WF2 No widows/orphans (1-word last lines; 1-line paragraph ends on new page).
- [ ] WF3 Never use "very".
- [ ] WF4 "In order to" → "To".
- [ ] WF5 Sort numeric citations: [2,5,7], not [7,2,5].
- [ ] WF6 Avoid brackets: unimportant ⇒ remove; important ⇒ not in brackets.
- [ ] WF7 No synonyms in scientific writing: one term per concept, used consistently.
- [ ] WF8 Avoid "performance" (ambiguous): say accuracy / speed / memory-use precisely.

## Tables & figures (WT)

- [ ] WT1 Captions self-contained ('comic-book mode' readers): explain how to read the
      figure AND end with the conclusion — what the reader should see ("So what?").
- [ ] WT2 Figures complete: all axes labelled with units, legend with clear differences,
      title per (sub)figure (no "(a)/(b)" explained only in the caption); lines/fonts big enough.
- [ ] WT3 Tables formatted per booktabs conventions (no vertical rules; \toprule/\midrule/
      \bottomrule).

## Introduction (WI)

- [ ] WI1 Funnel: start "just broad enough", narrow steadily, culminate at exactly our topic.
- [ ] WI2 No generic first sentence (if any paper in the field could open with it, cut it).
- [ ] WI3 End with explicitly bulleted contributions (~3); a contribution is something a peer
      researcher finds interesting.
- [ ] WI4 Figure 1 = visual abstract of the main idea (or pipeline).
- [ ] WI5 "Little research on X" is not a motivation — motivate by what is inherently
      interesting / who gains what; the gap itself goes in related work.

## Related work (WR)

- [ ] WR1 The METHOD is the subject, not the paper/authors ("X [a] is important, extended by
      Y [b]" — not "The work of [a] does X").
- [ ] WR2 One paragraph = one topic; 3–10 citations per paragraph.
- [ ] WR3 Paragraph layout: first sentence defines scope → group methods by what they do →
      concluding sentence relates them to OUR work ("all great, we use it" / "all great, but we
      differ because …").
- [ ] WR4 No history lessons: only work related to the research question; the section
      motivates what we build on and what we contrast with.

## Method (WM)

- [ ] WM1 No general argumentation — main-idea motivation lives in Intro/Related work; the
      method section motivates and explains only the technical method.
- [ ] WM2 No datasets in the method section (they belong in experiments).
- [ ] WM3 Number all equations.
- [ ] WM4 Equations are normal text: punctuate (period if sentence ends, comma if it continues).
- [ ] WM5 Explain ALL symbols directly before/after each equation; formula self-contained.
- [ ] WM6 Before an equation, first say in English why and what it achieves.
- [ ] WM7 The method must be understandable with all equations removed.
- [ ] WM8 Define only what is actually used later.

## Experiments (WE)

- [ ] WE1 Every experiment starts with an explicitly written question it answers.
- [ ] WE2 Group each experiment as a module (subsection/bold header), numbered:
      "Experiment 1: …".
- [ ] WE3 Modular analysis: each experiment has its own tables/figures; group what is compared
      together (repeating values across tables is fine).
- [ ] WE4 Experiment types in order: 1 Validate (does it do what we claim), 2 Investigate
      (unique properties), 3 Compare (vs others).
- [ ] WE5 Scale 0–1 scores to 0–100 ('0.07' → '7').
- [ ] WE6 Experiments do not "prove" — they demonstrate empirically for the setting at hand.
- [ ] WE7 Consider one more experiment for relevance in a different domain (or scope as
      future work).

## Discussion / conclusion (WD)

- [ ] WD1 Small summary of what was done, to set context.
- [ ] WD2 Limitations shown explicitly; insight into where it fails is strength.
- [ ] WD3 Conclusions answer "So what?" — modest and factual, but not shy about what is
      interesting and why.

## Course-template extras (CSE3000 appendix of the template)

- [ ] References: every in-text reference exists in the bibliography and vice versa; quotation
      marks + page numbers for literal quotes; paraphrases not too close to the original.
- [ ] Paragraph craft: clear topic sentences; clear argument line RQ → conclusions; literature
      reviewed critically.
- [ ] Style: objective; unambiguous; varied sentence length; active/passive mixed.
- [ ] Tables/figures: numbered + captioned, each referred to in text, interpretable standalone.
- [ ] Intro and conclusion understandable by any CS-bachelor reader.

## Project hard rules (override everything; from the thesis owner)

- [ ] No unsupported empirical claims — every empirical sentence has a measurement behind it.
- [ ] Interpretations explicitly labelled as interpretations.
- [ ] NEVER claim: "trained to convergence"; a novel methodology; in-domain validation as our
      contribution; that we fixed a field-wide confound.
- [ ] Budget framing: generous fixed budget + best in-domain-val checkpoint; disclose
      non-plateaued splits.
- [ ] Standard WSD components are cited, not invented.
- [ ] No invented citations or numbers; every reference verified against the primary source.
