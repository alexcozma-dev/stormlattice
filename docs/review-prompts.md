# Prompts for the Claude Code review (run from the repo root: cd ~/projects/stormlattice && claude)

Run these in order, in the same Claude Code session. Approve only the writes to the review files; decline any other edits.

---

## Pass 1: architecture review

Read docs/design.md (full design, the Nov 13 addendum and the decision log at the end) and README.md.
Do NOT edit anything. Write a critical review to docs/architecture-review.md.

Context: solo developer (engineering physics student, self-taught ML), submission
due Dec 2, judged Dec 3 on a 70-point rubric (demo video, live pitch, technical
feasibility, theme fit, usability, creativity, future work). Hackathon rules:
an LLM must play a real role; app hosted on GCP. Working so far: stormlattice/data/mrms.py
fetches, decodes and crops live MRMS radar to Florida, with offline and live tests.
Decisions I want kept unless you find a serious problem: trained model + physics
simulation as the core, LLM in a supporting role, SunTrax -> Orlando demo corridor.

Answer:
1. Feasibility: what's the minimum version that still makes the core claim
   (learned storm-birth rates beat advection)? What should be cut or deferred?
2. Weakest assumptions: where is the plan most likely to fail (data, labels,
   model skill, compute, demo day)? For each, an early test to find out fast.
3. Scientific soundness: problems with the cell tracking/labeling, the
   XGBoost birth/death framing, the lattice, or the evaluation.
4. Simpler alternatives that would give 80% of the result for 20% of the work.
5. Judging: what will impress non-expert judges, and what will they poke at?

Be direct. Rank findings by impact. Mark what's fine as-is.

---

## Pass 2: code structure review

Review the structure of this repo. Do NOT rewrite or edit files.
Write your findings to docs/structure-review.md.
Also read docs/architecture-review.md, and organize the code for the plan as it
would look after those findings.

Goals, in priority order:
1. Least convoluted path to a working demo by Dec 2 (solo developer, ~3 weeks).
2. Readability for a learner: I hand-type code to learn it, so prefer plain,
   explicit Python over clever abstractions.
3. Agent workability: easy for Claude Code to navigate, test and change safely.

Constraints: Python 3.10. Keep UTC everywhere. Don't change the science, only
the code organization. Use stormlattice/data/mrms.py and tests/test_mrms.py as
the reference for the style I'm using.

Check specifically:
- Is the package layout (stormlattice/<step>/...) right-sized, or should stubs
  be collapsed until they have real code?
- Imports: can scripts/ run directly? Should the package be pip-installable (-e)?
- Anything in scripts/ that belongs in the package (e.g. evaluate.csi).
- Config/settings approach, test layout, Docker/nginx files.
- Draft a CLAUDE.md with conventions for future agent sessions (include that I
  hand-type code: explain changes and propose small diffs, don't silently rewrite).

For each finding: the problem, why it matters, the smallest fix, and effort
(minutes). Rank by value. Flag anything that's fine as-is so I don't change it.
