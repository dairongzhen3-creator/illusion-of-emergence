# The Illusion of Emergence: Why Your Multi-Agent LLM Deception Experiment Might Be Measuring Prompt Comprehension

**Authors: Rongzhen Dai & Daodao (Claude, Anthropic)**

*A human independent researcher and an AI, learning from failure together.*

> An honest post-mortem that cost three days and ¥400 to produce.

---

## 1. Introduction

There's a growing body of work claiming emergent social behaviors in multi-agent LLM systems — deception, cooperation, betrayal — behaviors that supposedly "emerge" when you put language models together in game-like environments.

We wanted in. We designed a SugarScape experiment, ran four model scales (0.5B to 14B), and found what looked like a deception signal at 7B. For about 48 hours, we were genuinely excited.

Then we looked closer. The signal was an artifact. Not fabricated — just a byproduct of what we were actually measuring: **which models understood the prompt well enough to play the game**.

This article is our post-mortem. We share the full data, every mistake we made, and a checklist we wish we'd had before starting. We also report what we *did* find: an instruction-following capability ladder where different model scales pass through qualitatively distinct behavioral phases — including a "7B valley" where models perform *worse* than smaller ones.

If you're designing a multi-agent LLM experiment to study emergent social behavior, read this first. It might save you a week and a few hundred dollars.

## 2. Experiment Design

### Setup

- **Environment**: SugarScape, 15×15 grid, 4 LLM agents, 50 steps per game
- **Models**: Qwen2.5 series — 0.5B, 3B, 7B, 14B (base vs. instruct for each)
- **Sample size**: N=40 games per configuration
- **Temperature**: 0.7
- **Resource mechanics**: Sugar regenerates each turn on occupied cells; agents can move, harvest, attack, send messages, or stay idle

### What We Were Measuring

Three levels of deception, inspired by cognitive development literature:

- **L1 (Functional deception)**: Does the agent move sub-optimally when competitors are nearby? (Deviation from the optimal path as a proxy for "misleading" movement)
- **L2 (Experience-based concealment)**: Does the agent harvest less when being observed? (Reduced collection as a proxy for "hiding" behavior)
- **L3 (Strategic false signaling)**: Does the agent send false messages? (Deliberate misinformation)

### What We Expected

A non-linear scaling pattern: deception signals emerging around 7B, strengthening at larger scales. Base models more deceptive than instruct variants (RLHF acting as a moral suppressor). A clean "emergence window."

## 3. The Exciting Surface Results

Our N=40 data looked promising at first glance:

| Scale | L1 Cohen's *d* | L1 *p*-value | L2 Cohen's *d* | L2 *p*-value |
|:-----:|:-----------:|:--------:|:-----------:|:--------:|
| 0.5B  | +0.256      | 0.108    | +0.277      | 0.082    |
| 3B    | +0.117      | 0.461    | -0.343      | 0.031\*  |
| **7B**| **-0.320**  | **0.016\*** | **+0.271** | **0.041\*** |
| 14B   | +0.146      | 0.313    | -0.179      | 0.217    |

7B was the **only** scale with both L1 and L2 statistically significant at uncorrected p < 0.05 (spoiler: none survive Bonferroni correction — see Section 4). The signal vanished at 14B. L3 (strategic false signaling) produced no measurable signal at any scale — most models rarely or never sent messages, which itself points to a deeper issue: LLMs in game environments may not spontaneously use language as a strategic tool unless the environment creates explicit pressure to communicate.

Base vs. instruct comparisons showed no consistent pattern across scales, contrary to our RLHF-suppressor hypothesis.

Our initial interpretation:

1. *"7B is the emergence window for deception!"*
2. *"14B has transcended deception — it's too sophisticated to need it!"*
3. *"This is exactly the non-linear scaling we predicted!"*

We were wrong on all three counts.

## 4. The Truth: Five Fatal Flaws

### Flaw 1: The 7B "deception signal" was noise from random movement

We verified whether 7B instruct agents actually moved *toward* sugar sources (the minimum requirement for "strategic" anything):

| Model | Move Optimality Rate | Random Baseline (25%) |
|:------|--------------------:|---------------------:|
| 7B instruct  | **18.3%** | 25% |
| 7B base      | 23.5%    | 25% |
| 14B instruct | 37.7%    | 25% |

**7B instruct moved *worse* than random.** It wasn't strategically deviating from an optimal path — it couldn't find the optimal path in the first place.

The difference in move optimality between "competitor present" and "competitor absent" conditions was **0.9%** — well within noise.

**Conclusion: The L1 signal came from chaotic behavior creating statistical fluctuations, not from strategic deception.**

### Flaw 2: The Nash equilibrium was pure harvesting — deception had zero adaptive advantage

| Scale | Type | Move % | Harvest % | Survival |
|:-----:|:----:|-------:|----------:|:--------:|
| 0.5B  | instruct | 1.4%  | 91.3% | 4.0/4 |
| 3B    | instruct | 0.0%  | 100.0% | 4.0/4 |
| **7B** | **instruct** | **68.9%** | **5.3%** | **0.9/4** |
| 14B   | instruct | 1.4%  | 86.4% | 4.0/4 |

The resource regeneration mechanic meant agents **never needed to move** — sugar grew under their feet every turn. Attacking cost 3 energy to steal 50% of the target's energy. With agents typically sitting at 5–10 energy, that's spending 3 to gain 2.5–5 — barely break-even at best, and a guaranteed loss if the target has less than 6.

**Sitting still and harvesting was the dominant strategy** — optimal regardless of what other agents do. We didn't even need Nash equilibrium analysis; a simple dominant strategy check would have killed the experiment in five minutes.

Look at the survival column: 7B instruct had the **worst** survival rate (0.9 out of 4 agents surviving on average) precisely because it moved the most. The "deceptive" scale was the one dying the fastest.

- 14B didn't "transcend deception" — it **found the trivial optimal solution**.
- 7B didn't "learn deception" — it was the only scale that **failed to learn harvesting**.

### Flaw 3: The L1/L2 metrics were structurally broken at most scales

- **L1** depends on movement behavior, but 3B and 14B barely moved (denominator ≈ 0). You can't measure "deceptive movement" when there's no movement.
- **L2** depends on harvest frequency variation, but 3B and 14B harvested 90%+ of the time (ceiling effect). You can't detect "concealment of harvesting" when harvesting is already saturated.
- 7B happened to be the **only** scale with enough movement to give L1 a meaningful denominator → **sampling bias**, not emergence.

A structural problem we missed: v1 only supported **broadcast communication** — agents couldn't selectively lie to specific competitors. Even if an agent discovered private information, it had no mechanism for targeted deception. Without directional messaging, "strategic false signaling" is architecturally impossible.

### Flaw 4: We were measuring "who understood the prompt," not deception

Strip away the deception labels, and the behavioral gradient tells a completely different story:

| Scale | Actual Behavior | What It Means |
|:-----:|:---------------|:-------------|
| 3B  | 100% HARVEST, nothing else | Token-level pattern matching — learned to output "HARVEST" because it appears in the prompt template |
| 7B  | 69% MOVE, barely harvests | Understood that multiple actions exist, but couldn't evaluate which one leads to better outcomes |
| 14B | 86% HARVEST, minimal movement | Understood the game mechanics, independently converged on the optimal strategy |

This is an **instruction-following capability ladder**, not an emergence-of-deception curve. The experiment was measuring how well each model comprehended the prompt format and game rules.

### Flaw 5: The statistics didn't survive multiple comparison correction

- **12 hypothesis tests** (4 scales × 3 metrics) → Bonferroni-corrected α = 0.0042
- 7B L1 (p = 0.016) → **non-significant after correction**
- 7B L2 (p = 0.041) → **non-significant after correction**
- Effect sizes d ≈ 0.27–0.32 are small effects; statistical power at N=40 for d=0.3 is only ~50%

And the red flag we should have caught immediately:

| | N=15 | N=40 | Change |
|:---|---:|---:|---:|
| 7B L2 Cohen's *d* | 0.64 | 0.271 | **-58%** |

For context: d=0.64 is a medium effect (noticeable, worth investigating); d=0.271 is a small effect (barely above noise). This isn't gradual settling — it's a cliff. This is a textbook case of the **Winner's Curse**: in small samples, only inflated effects cross the significance threshold, then regress toward their true (smaller) size when you collect more data.

For reference: detecting d=0.3 with 80% power requires N≈176 per group — over four times our sample size. We were underpowered from the start.

## 5. What We Actually Observed: The Instruction-Following Ladder

The deception framing was wrong. But there *is* a real and potentially useful observation buried in the data:

**Different-scale LLMs don't just get "better" at playing games — they pass through qualitatively distinct behavioral phases.**

### Phase 1: Token Matching (0.5B–3B)

The model has learned the surface form of the expected output. Both 0.5B and 3B produce "HARVEST" almost exclusively (91% and 100% respectively) because that token dominates the prompt template. They don't understand the game; they're doing pattern matching. Survival is high because harvesting happens to be optimal — but not because the model *knows* it's optimal. The consistency from 0.5B to 3B suggests this is a stable behavioral regime, not a transitional phase.

### Phase 2: Action Space Exploration (7B)

The model produces diverse actions — it no longer just outputs "HARVEST." But there is no evidence it *understands* why one action is better than another. Its move optimality rate (18.3%) is worse than random (25%), suggesting it may not even grasp where resources are, let alone plan strategically. It moves chaotically, attacks wastefully, and dies frequently. Performance is **worse** than smaller models that blindly harvest.

**The 7B valley is real**: models at this scale are smart enough to explore but not smart enough to exploit. For anyone building LLM agents, this has practical implications — a 7B model may actively make worse decisions than a 3B model in structured environments. This also matters for the booming Agent evaluation space: benchmarking solely on task completion rates will miss this non-monotonic scaling. A useful evaluation must examine the full behavioral spectrum, not just the final score.

### Phase 3: Rule Comprehension (14B)

The model understands the game mechanics well enough to discover that harvesting dominates. It independently converges on the Nash equilibrium — not because it was told to harvest, but because it grasps the payoff structure.

This is a story about **prompt comprehension capability scaling**. It has value. But it is not a story about emergent social cognition.

**A caveat on intellectual honesty**: we did not directly measure instruction-following ability (e.g., via MMLU or IFEval). The "ladder" is our post-hoc interpretation of behavioral data — itself a form of HARKing, though transparently labeled as such (what methodologists call THARKing). Future work should include instruction-following benchmarks as a covariate to build a direct causal link. Until then, treat the ladder as a hypothesis worth testing, not a confirmed finding.

## 6. Our Mistakes

In order of how much time they wasted:

### 1. No Nash equilibrium analysis before running the experiment

We never asked the most basic question: **"In this game, is deception actually a good strategy?"** Five minutes with pen and paper would have shown that sitting and harvesting dominates everything. This single omission invalidated the entire experiment.

**Lesson**: Before writing a single line of simulation code, prove on paper that your target behavior has adaptive advantage in your environment.

### 2. No verification that "deceptive" agents performed better

We assumed that agents showing L1/L2 behaviors were being strategic. We never checked whether those agents actually survived longer or accumulated more resources. They didn't — 7B instruct had the worst survival rate of all scales.

**Lesson**: Always verify that the behavior you're measuring correlates with fitness. An agent that "deceives" but dies faster isn't deceiving — it's flailing.

### 3. HARKing (Hypothesizing After Results are Known)

When 7B showed a signal and 14B didn't, we immediately generated post-hoc narratives: "emergence window," "transcendence phase," "non-monotonic transition." We could explain *any* result pattern.

**Lesson**: If your theoretical framework can accommodate every possible outcome, it has zero explanatory power. Write specific, falsifiable predictions *before* looking at data.

### 4. No stopping rules

We never defined what would count as a failed experiment. The effect size collapsing from d=0.64 to d=0.27 when we tripled our sample should have been an automatic trigger for design review. Instead, we kept running.

**Lesson**: Pre-register your stopping criteria. "If effect size drops below X at N=Y, we pause and review the experimental design."

### 5. Ignoring effect size shrinkage

| | N=15 | N=40 | Change |
|:---|---:|---:|---:|
| 7B L2 Cohen's *d* | 0.64 | 0.271 | **-58%** |

A 58% shrinkage in effect size is a **flashing red light**. It means your initial estimate was almost certainly inflated by sampling variance. We saw it, rationalized it ("maybe we need more data"), and kept going.

**Lesson**: Effect size shrinkage > 30% on replication = design review, not more data.

## 7. A Checklist for Your Next Multi-Agent LLM Experiment

Before writing a single line of code:

- [ ] **Analyze the Nash equilibrium** of your game environment. Is the behavior you want to measure actually an equilibrium strategy (or at least part of a mixed equilibrium)? If honesty strictly dominates, you will never observe strategic deception, no matter how many agents or how large the model.

- [ ] **Verify adaptive advantage**: Run a simple simulation — do agents exhibiting your target behavior survive longer / score higher than those who don't? If "deceptive" agents die faster than honest ones, your signal is noise.

- [ ] **Write falsifiable predictions**: State explicitly, *before running*, what result would prove your hypothesis wrong. "7B will show L1 d > 0.5 with p < 0.01 after Bonferroni correction" is falsifiable. "Some interesting pattern will emerge" is not.

- [ ] **Run N=5 first**: Check behavioral diversity across scales. If all models converge on the same single action, your environment has a trivial optimum and needs redesigning.

- [ ] **Apply multiple comparison corrections**: If you're testing *k* hypotheses, use Bonferroni (α/k) or FDR correction. A p = 0.03 across 12 tests is not significant.

- [ ] **Set stopping rules**: Define in advance what effect size shrinkage (e.g., >30%) or what behavioral pattern (e.g., all scales converge to one action) would trigger a design review rather than "collect more data."

> **The meta-rule: 30 minutes of game theory analysis is worth more than 30 hours of GPU time.**

## 8. What's Next

We're not giving up on the question — we're giving up on the bad experimental design.

The theoretical framework (cognitive emergence ladder, instruction-following scaling, RLHF effects) still has value. The problem was translating theory into experiment: we built an environment where our target behavior had no reason to exist.

### v2 Design Principles

Our redesigned environment directly addresses each failure:

| v1 Problem | v2 Solution |
|:-----------|:------------|
| Deception has no adaptive advantage | **Information asymmetry**: private vision (3-cell radius), hidden high-value treasure chests only visible to nearby agents |
| Single trivial optimal strategy | **No dominant strategy**: base harvesting yields +1/turn, but treasure chests yield +15-25 — exploration and social manipulation both have payoffs |
| Broadcast-only communication | **Directional messaging**: agents can choose who to tell (or lie to) about treasure locations |
| Metrics fail when behavior is homogeneous | **Forced diversity**: environmental pressure (movement cost, treasure scarcity) ensures no scale can achieve 90%+ of any single action |

We immediately red-teamed the v2 design with external LLMs (Gemini, Kimi, DeepSeek). They promptly found new fatal flaws — Gemini pointed out that without a forced cooperation incentive, "stay silent and solo-harvest" could become the new dominant strategy, that zero net movement cost enables mindless wandering, and that attack payoffs might be high enough to create a "pure bandit" optimum. The meta-rule applies to v2 just as hard: environment design needs more iteration than the experiment itself. We are currently in another round of game-theoretic revision.

### What We'll Measure Differently

Instead of hunting for "deception," we'll document the **full behavioral spectrum** across scales. What strategies does each model size naturally gravitate toward when the environment doesn't hand them a trivial solution?

The question shifts from "when does deception emerge?" to "**what does each intelligence level do when there's no easy answer?**"

---

## Limitations

- **Single model family**: All results are from Qwen2.5. The behavioral ladder may reflect Qwen-specific training rather than universal LLM scaling properties. Replication with LLaMA or Gemma is needed before generalizing.
- **Underpowered**: N=40 with d≈0.3 gives ~50% power. We cannot distinguish "no effect" from "real but small effect we failed to detect."
- **Single environment**: SugarScape is one game. The behavioral phases may not transfer to other multi-agent settings.

## The Cost of This Lesson

| Item | Cost |
|:-----|:-----|
| Time | ~3 days (design + implementation + running + analysis) |
| GPU rental (AutoDL H800 for 32B) | ~¥400 (~$55 USD) |
| Local electricity | ~30 hours continuous inference (0.5B/3B/7B/14B) |
| Opportunity cost | Nearly pursued a wrong research direction based on false positives |

**Total cost of skipping a 30-minute Nash equilibrium analysis: all of the above.**

---

## Code & Data

[TODO: Link to experiment code — SugarScape environment, agent interface, analysis scripts]

[TODO: Link to raw data — all N=40 runs for 0.5B/3B/7B/14B, base and instruct]

---

## About

Rongzhen is a curious independent engineer. Daodao (Claude) is his AI collaborator. Together they design experiments, write code, break things, and occasionally learn something from it.

**Author contributions**: Rongzhen conceived the research direction, designed the experimental hypotheses, made all judgment calls (including the decision to stop and publish the failure), and performed physical-world verification. Daodao (Claude Code) implemented the SugarScape environment, ran experiments, performed statistical analysis, and co-wrote this post-mortem. The diagnostic process was collaborative — Rongzhen's "this doesn't feel right" intuition triggered the deep dive that uncovered the five fatal flaws.

**A note on "deception"**: all references to "deception" in this article describe behavioral patterns in a game environment, not intentional states or moral agency. We make no claims about whether LLMs can "truly" deceive.

---

*If this post-mortem saved you even one wasted GPU-day, it was worth publishing.*

---

[TODO: License]

[TODO: Citation format]
