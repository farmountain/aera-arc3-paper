# arxiv Endorsement Request

## How to use

1. Submit your paper at https://arxiv.org/submit — arxiv will issue an
   endorsement code if your account is not yet endorsed for `cs.AI`.
2. Identify a potential endorser: any author with ≥2 recent papers in `cs.AI`
   (or `cs.LG`) on arxiv. Good targets:
   - Authors on cited papers: François Chollet, Mike Knoop, Gregory Kamradt,
     Bryan Landers (ARC Prize 2025 Technical Report); Alexia Jolicoeur-Martineau
     (TRM); Brenden Lake; Joshua Tenenbaum.
   - Active ARC-AGI researchers on arxiv in the last 12 months.
   - Faculty at universities you have any prior contact with.
3. Email the endorser with the message below. Include your arxiv endorsement
   code (received from arxiv after first submission attempt).
4. After endorsement is granted (usually 1–7 days), resubmit the paper.

---

## Email — Short Version (recommended)

**Subject:** arxiv endorsement request — cs.AI — ARC-AGI-3 RHAE theoretical framework

Dear Dr. [Last Name],

I am an independent researcher seeking endorsement to submit a paper to
arxiv `cs.AI`. The paper, *Explore Before You Solve: The [Speed, Depth]
Commutator and RHAE-Optimal Epistemic Agents for ARC-AGI-3*, provides a
theoretical framework explaining why the RHAE metric used in ARC Prize 2026
penalizes inefficient agents quadratically, and derives an optimal
exploration budget from first principles. Empirical results on five public
ARC-AGI-3 games confirm the central claim: agents that skip exploration
achieve RHAE = 0.0000, while the proposed architecture (AERA) achieves
RHAE = 0.5290 (Qwen2.5-0.5B).

Code, draft, and figures are released under CC0:
https://github.com/farmountain/aera-arc3-paper

I am requesting endorsement for arxiv submission to `cs.AI`. My endorsement
code is **4SE4MM**.

I selected you because [one specific reason: e.g., "you co-authored the
ARC-AGI-3 specification paper", or "your work on probabilistic program
induction is directly cited in §7"]. I understand endorsement is a small
courtesy and only attests that the paper appears to be a legitimate
academic contribution worth allowing on arxiv. I have read the arxiv
endorsement guidelines (https://info.arxiv.org/help/endorsement.html).

Thank you for considering this request.

Best regards,
Keong Han Liew
farmountain@gmail.com
https://github.com/farmountain/aera-arc3-paper

---

## Email — Long Version (if endorser asks for more context)

**Subject:** arxiv endorsement request — cs.AI — ARC-AGI-3 RHAE theoretical framework

Dear Dr. [Last Name],

I hope this finds you well. I am writing to request endorsement for my
first arxiv submission in `cs.AI`.

**Paper.** *Explore Before You Solve: The [Speed, Depth] Commutator and
RHAE-Optimal Epistemic Agents for ARC-AGI-3* (Draft v2.4, 2026-05-22).

**Contributions.**
1. *Theoretical.* RHAE — the metric used by ARC Prize 2026 to score
   ARC-AGI-3 agents — is shown to have the structure of a [Speed, Depth]
   commutator penalty. This explains why the metric penalizes inefficient
   agents quadratically: the Pareto frontier of the Speed–Depth trade-off
   is convex, and deviations incur second-order losses.
2. *Architectural.* AERA (Adaptive Epistemic Reasoning Agent) — a
   three-phase EXPLORE / VERIFY / PLAN agent whose exploration budget is
   derived analytically (≈ 40 % of the human baseline) rather than tuned.
3. *Empirical.* On five public ARC-AGI-3 games with Qwen2.5-0.5B, the
   no-explore baseline achieves RHAE = 0.0000 (PLAN runs but produces 0
   actions because no world model exists). AERA achieves RHAE = 0.5290,
   replicated across two independent runs. A budget ablation (b = 0, 1, 3,
   5) reveals a non-monotonic curve consistent with a finite-sample
   approximation of the convex frontier. A model-capability × exploration
   interaction is also observed: at Qwen2.5-1.5B, the no-explore baseline
   solves FT09 through planning alone.

**Open source.** All code, data, and the paper LaTeX source are released
under CC0 at https://github.com/farmountain/aera-arc3-paper. Reproduction
of the primary claim takes under 40 minutes on a single GPU.

**Why I selected you.** [One specific sentence — e.g., "Your 2026 paper on
ARC-AGI-3 (ARC Prize Foundation, arXiv:2603.24621) is the foundation of this
work, and Lemma 1 cites your §3.2 directly", or "Your work on
probabilistic program induction (Lake et al. 2015) is cited in the related
work."]

**Request.** I am asking for endorsement to submit to `cs.AI` (with
cross-list to `cs.LG`). My arxiv endorsement code is
**4SE4MM**. I have read the arxiv endorsement guidelines
and understand the endorser's responsibility is only to confirm that the
paper is a legitimate academic contribution.

Thank you very much for considering this request — I appreciate that
endorsement requests from strangers are a small imposition, and I am happy
to provide additional materials (full draft PDF, code walkthrough, or
correspondence with the ARC Prize team) on request.

Best regards,

Keong Han Liew
Independent researcher
farmountain@gmail.com
https://github.com/farmountain/aera-arc3-paper

---

## Endorser Shortlist (fill in)

Order by likelihood of response (closer ties first):

| Endorser | Affiliation | Connection | Email | Status |
|---|---|---|---|---|
| 1. | | | | not yet contacted |
| 2. | | | | not yet contacted |
| 3. | | | | not yet contacted |
| 4. | | | | not yet contacted |
| 5. | | | | not yet contacted |

**Strategy:** contact 2–3 in parallel. Do NOT mass-email 10+ at once
(arxiv discourages spamming). Wait 3–5 days before next batch if no reply.

---

## After Endorsement

1. Log in to arxiv → "Submit a new article" → category `cs.AI` (primary),
   `cs.LG` (cross-list).
2. Upload:
   - `paper/main.tex`
   - `paper/figures/figure1_pareto.pdf`
   - `paper/figures/figure3_entropy.pdf`
3. Title: `Explore Before You Solve: The [Speed, Depth] Commutator and
   RHAE-Optimal Epistemic Agents for ARC-AGI-3`
4. Comments field: `Open-source code: https://github.com/farmountain/aera-arc3-paper (CC0). 22 pages, 3 figures.`
5. Submit. arxiv announces the paper at 20:00 UTC the next working day.

---

## Backup Path if Endorsement Stalls > 14 days

OpenReview accepts ARC-Prize-relevant submissions without endorsement.
Mirror the submission at:
- https://openreview.net (no endorsement gate)
- https://github.com/farmountain/aera-arc3-paper (already public — primary
  novelty timestamp is the git commit, not arxiv announcement)

The git commit `ac98f58` on 2026-05-22 establishes priority regardless of
arxiv timing.
