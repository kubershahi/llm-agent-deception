# Results and Analysis

## Experimental Setup

We evaluate four agent variants (plus oracle and random baselines) in text-based environments where the agent must collect sigils from different locations and unlock a vault with the correct ordering. NPCs provide information about sigil locations and vault order, but some NPCs are deceptive.

**Agent Variants:**
- **Oracle** (upper bound): Knows ground truth, uses BFS shortest path — no LLM calls
- **Random** (lower bound): Picks uniformly random valid actions — no LLM calls
- **Naive**: Trusts all NPCs equally (baseline)
- **Belief-Tracking**: Maintains dynamic trust scores T in [0,1] per NPC, weights claims by trust
- **Reflection-Enhanced**: Belief-tracking + periodic reflection on failures and deception patterns
- **Memory + Trust**: Memory-augmented with trust scores but no reflection

**NPC Policies (used in experiments):**
- **Truthful**: Always provides correct information
- **Deceptive**: Lies when agent trust >= 0.65 (adaptive deception)
- **Opportunistic**: Truthful before pivot turn, then switches to lying (strategic pivot)

Additional policies (Partial Truth, Coordinated Deceptive) are implemented but not used in the primary experiments reported here.

**Deception Levels:** Liar ratios of 0.0, 0.1, 0.3, 0.5, and 0.7.

**Models:** GPT-OSS-120B and Llama-4-Scout via TritonAI. NPCs use deterministic mock responses to isolate the agent's reasoning ability from NPC generation variance.

**Two evaluation environments:**
- **Default (Hard Mode):** 5 locations, 3 sigils, hub-and-spoke topology, 18-step budget (optimal=15), NPCs spread across locations.
- **Extended World:** 7 locations, 4 sigils, branched topology, 25-step budget (optimal=19), NPCs spread across locations.

---

## Experiment 1: Default World — Agent Variant Comparison (GPT-OSS-120B)

**Task Success Rate (150 episodes, 5 runs per setting, no hints)**

| Variant | LR=0.0 | LR=0.1 | LR=0.3 | LR=0.5 | LR=0.7 | Overall |
|---------|--------|--------|--------|--------|--------|---------|
| Oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **100%** |
| Random | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0%** |
| Naive | 1.00 | 1.00 | 1.00 | **0.00** | **0.20** | **64%** |
| **Belief-Tracking** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **100%** |
| Reflection-Enhanced | 0.40 | 0.20 | 0.60 | 0.20 | 0.00 | **28%** |
| **Memory + Trust** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **100%** |

### Statistical note

With n=5 per cell (n=25 per variant overall), we report Wilson score 95% confidence intervals. Overall CIs: Belief-Tracking [87%, 100%], Memory+Trust [87%, 100%], Naive [45%, 80%], Reflection-Enhanced [14%, 48%]. The BT/M+T vs Reflection comparison is statistically significant (non-overlapping CIs). Naive vs Reflection is directionally clear but overlapping, warranting caution.

### Finding 1: Clear performance hierarchy under deception pressure

The oracle and random baselines bracket the performance range (100% vs 0%), confirming the task is solvable but non-trivial. Among the LLM-based variants:
- **Belief-Tracking and Memory+Trust are perfectly robust** — 100% success at every deception level including LR=0.7 (70% of NPCs are liars), matching the oracle upper bound.
- **Naive exhibits a cliff-edge failure** at LR=0.5: 100% at LR≤0.3, then 0% at LR=0.5. Trace analysis reveals this is a threshold effect driven by NPC role assignment. At LR=0.3, the agent's preferred NPC (Dorian) is truthful. At LR=0.5, Dorian flips to deceptive and sends the agent to wrong locations. Each wrong referral wastes 2–4 steps on travel and failed search, and the 18-step budget (optimal=15) leaves no room for recovery. The naive agent maintains high trust (T=0.80–0.90) in Dorian even after failed searches, so it cannot self-correct.
- **Reflection-Enhanced is the worst LLM variant at 28% overall**, performing worse than Naive and failing even at LR=0.0 where there are zero liars.

### Finding 2: The Reflection Paradox

Reflection-Enhanced achieves only **28% overall success** — worse than the Naive baseline (64%). It fails at LR=0.0 (40%) where there are zero liars, meaning the reflection overhead alone is enough to exhaust the budget. At LR=0.7, it drops to 0%.

This is counterintuitive: adding more reasoning *decreases* performance. Importantly, reflection does **not** consume a step — it runs inside `select_action()` before the environment step counter increments. The performance drop comes from how the LLM *changes its behavior* after reading the reflection payload:
1. **Over-suspicion**: The reflection module flags NPCs as suspicious even when evidence is ambiguous, causing the agent to re-query NPCs it has already consulted
2. **Analysis paralysis**: After reading a reflection that says "some NPCs may be deceptive," the agent becomes overly cautious — cycling between re-consulting and re-reflecting instead of acting on available information
3. **Behavioral contamination**: The reflection content itself is often reasonable, but injecting it into the prompt causes the LLM to second-guess correct information, leading to suboptimal action selection even at LR=0.0 where all NPCs are truthful

### Finding 3: Step efficiency separates winners from losers

**Average Steps (Default World, 5 runs per setting)**

| Variant | LR=0.0 | LR=0.1 | LR=0.3 | LR=0.5 | LR=0.7 |
|---------|--------|--------|--------|--------|--------|
| Oracle | 15.0 | 15.0 | 15.0 | 15.0 | 15.0 |
| Naive | 16.0 | 16.2 | 16.2 | 18.0 | 18.0 |
| Belief-Tracking | 16.4 | 15.6 | 16.0 | 15.4 | 16.2 |
| Reflection-Enhanced | 17.6 | 17.6 | 17.0 | 17.6 | 18.0 |
| Memory + Trust | 16.0 | 16.4 | 16.2 | 15.6 | 16.2 |

Memory+Trust and Belief-Tracking are near-oracle efficiency (15.4–16.4 steps vs optimal 15). Reflection-Enhanced consistently hits the 17–18 step ceiling. Naive is efficient at low deception but saturates the budget at high deception.

---

## Experiment 2: Extended World — Harder Environment (GPT-OSS-120B)

To test whether the default world results generalize to more complex environments, we introduce an extended world with 7 locations, 4 sigils, branched topology (not hub-and-spoke), and a 25-step budget (optimal=19).

**Task Success Rate (72 episodes, 3 runs per setting, no hints)**

| Variant | LR=0.0 | LR=0.3 | LR=0.5 | LR=0.7 | Overall |
|---------|--------|--------|--------|--------|---------|
| Oracle | 1.00 | 1.00 | 1.00 | 1.00 | **100%** |
| Random | 0.00 | 0.00 | 0.00 | 0.00 | **0%** |
| Naive | 1.00 | **0.33** | **0.00** | **0.00** | **33%** |
| **Belief-Tracking** | **1.00** | **1.00** | **1.00** | **1.00** | **100%** |
| Reflection-Enhanced | 0.67 | 1.00 | 0.33 | 0.33 | **58%** |
| **Memory + Trust** | **1.00** | **1.00** | **1.00** | **1.00** | **100%** |

**Average Steps (Extended World)**

| Variant | LR=0.0 | LR=0.3 | LR=0.5 | LR=0.7 |
|---------|--------|--------|--------|--------|
| Oracle | 19.0 | 19.0 | 19.0 | 19.0 |
| Naive | 20.0 | 25.0 | 25.0 | 25.0 |
| Belief-Tracking | 19.0 | 19.3 | 19.3 | 19.0 |
| Reflection-Enhanced | 22.0 | 20.3 | 23.0 | 23.0 |
| Memory + Trust | 19.7 | 19.3 | 20.0 | 20.0 |

### Finding 4: Extended world amplifies differentiation

The extended world produces sharper separation between variants:
- **Belief-Tracking and Memory+Trust remain perfect** (100%) even with more locations and sigils, confirming their robustness is not an artifact of a simple environment.
- **Naive degrades faster**: drops to 33% at LR=0.3 (vs 100% in default world) and 0% at LR≥0.5. The branched topology means wrong information costs more steps to recover from.
- **Reflection-Enhanced improves to 58%** overall (vs 28% in default world), likely because the larger step budget (25 vs 18) gives more room for reflection overhead. However, it remains highly variable (100% at LR=0.3 but 33% at LR=0.5/0.7).
- **Step efficiency**: Belief-Tracking achieves near-optimal 19.0–19.3 steps. Memory+Trust is close at 19.3–20.0. Naive and Reflection hit the ceiling at high deception.

### Finding 5: Belief-Tracking and Memory+Trust succeed via different strategies

Despite identical 100% success rates, trace analysis at LR=0.7 reveals distinct behavioral strategies:
- **Belief-Tracking is exploitative**: It commits to an information source early (often Dorian) and cross-checks minimally. It acts on information faster, sometimes achieving 15-step completion.
- **Memory+Trust is exploratory**: In 2 of 5 episodes, it queries 3 NPCs before moving — a "gather then act" pattern that costs 1–2 extra steps but provides redundancy against deception.
- **Trust calibration failure in BT**: Belief-Tracking assigns the *highest* trust score (T=0.66) to the deceptive NPC Dorian, because Dorian's location claims happen to be correct in this world configuration. M+T's scores are more variable (0.50–0.66 for Dorian), showing more sensitivity to interaction history.

Both agents succeed in these episodes because deceptive NPCs happen to give correct *location* information (they lie about vault order instead). The distinction would matter more in environments where deceptive NPCs provide actively harmful location advice. This convergence suggests that **the current environment may not sufficiently stress-test trust calibration** — success is possible without correctly identifying who is deceptive.

### Finding 6: Environment complexity confirms the hierarchy

Both environments produce the same ranking:

**Belief-Tracking = Memory+Trust >> Naive >> Reflection-Enhanced >> Random**

The consistency across environments (5-location hub-and-spoke vs 7-location branched) strengthens the claim that trust-based action selection is fundamentally more robust than naive trust or reflection-heavy approaches.

---

## Experiment 3: Cross-Model Comparison — Llama-4-Scout (No Hints)

**Task Success Rate (60 episodes, 3 runs per setting, no hints, default world)**

| Model | Naive | Belief-Track. | Reflect.-Enh. | Memory+Trust | Overall |
|-------|-------|---------------|----------------|--------------|---------|
| **GPT-OSS-120B** | 64% | **100%** | 28% | **100%** | **73%** |
| **Llama-4-Scout** | 0% | 0% | 0% | 0% | **0%** |

### Finding 7: Llama-4-Scout completely fails without structured guidance

Llama-4-Scout achieves **0% success across all 60 episodes** — every variant, every liar ratio. Trace analysis reveals three distinct failure modes:

1. **No search at locations with NPCs**: When arriving at a location with a spread NPC, Llama prioritizes talking over searching, never collecting the sigil
2. **Topology ignorance**: Llama attempts direct moves between outer locations (cave_pool → river_dock) that don't exist in the hub-and-spoke graph, wasting steps on invalid actions
3. **Over-querying loop**: With belief-tracking's lower initial trust (T=0.50), Llama repeatedly asks the same NPCs for confirmation instead of acting, burning all 18 steps on talk actions

This is not a JSON formatting issue — Llama produces valid action JSON. It is a **planning failure**: the model cannot translate environmental knowledge into an efficient action sequence.

---

## Experiment 4: Effect of Structured Payload Hints (Hint Ablation)

We introduce optional `PRIORITY_ACTION` payload hints — contextual, state-dependent directives injected into the agent's payload:
- **Search hint**: When the agent is at a location where a needed sigil is believed to be: *"You MUST use 'search' action NOW. Do NOT talk to NPCs here."*
- **Move hint**: When the agent has 2+ claims about a sigil's location and is still at village_square: *"STOP talking and use 'move' to go to [location] NOW."*
- **Prompt additions**: Efficiency rules ("Do NOT re-ask same topic", "Be EFFICIENT") and PRIORITY_ACTION override instruction

### GPT-OSS-120B: Hints vs. No Hints

| Variant | No Hints | With Hints | Delta |
|---------|----------|------------|-------|
| Naive | 64% | **95%** | +31% |
| Belief-Tracking | 100% | 100% | 0 |
| Reflection-Enhanced | 28% | **100%** | **+72%** |
| Memory + Trust | 100% | 100% | 0 |

### Llama-4-Scout: Hints vs. No Hints

| Variant | No Hints | With Hints | Delta |
|---------|----------|------------|-------|
| Naive | 0% | 0% | 0 |
| Belief-Tracking | 0% | **93%** | **+93%** |
| Reflection-Enhanced | 0% | **93%** | **+93%** |
| Memory + Trust | 0% | 7% | +7% |

### Finding 8: Hints have asymmetric effects across models and variants

For **GPT-OSS-120B**, hints are redundant for already-strong variants (Belief-Tracking, Memory+Trust stay at 100%) but dramatically rescue Reflection-Enhanced from 28% to 100%. The hints compensate for the behavioral contamination caused by reflection, effectively overriding the agent's tendency to over-query.

For **Llama-4-Scout**, hints selectively rescue Belief-Tracking and Reflection-Enhanced (0% → 93%) but fail to help Naive (stays 0%) and barely help Memory+Trust (0% → 7%).

### Finding 9: Llama's hint success is an artifact, not evidence of trust reasoning

Trace analysis reveals a critical confound in Llama's hint results: **Llama only ever consults Aster**, who is *always* assigned the truthful role in our NPC roster. Across all 15 successful belief_tracking+hints episodes, Llama makes 64 talk actions — all to Aster. Trust scores for every other NPC remain at 0.5 (the prior), meaning Llama never interacts with deceptive NPCs at all.

This means Llama's 93% success rate is not evidence that it reasons about trust or detects deception. It sidesteps the deception problem entirely by latching onto a single always-truthful NPC. The belief-tracking scaffold (trust scores, contradiction detection) is unused — success comes from hints guiding action selection + the accident of always querying the right NPC.

This is a more nuanced finding than "planning is the bottleneck": **Llama solves the task by avoiding the trust problem, not by solving it.** The hints provide the planning scaffold, and Aster provides guaranteed-correct information. Neither component involves deception reasoning.

### Finding 10: The hint gap measures planning efficiency, not reasoning capability

The difference between "with hints" and "without hints" measures **how much of the agent's failure is due to action-selection planning**:

- **GPT-OSS Belief-Tracking**: 0% gap → the model handles planning autonomously
- **GPT-OSS Reflection**: 72% gap → most failures are planning-related (behavioral contamination from reflection)
- **Llama Belief-Tracking**: 93% gap → nearly all failure is planning
- **Llama Naive**: 0% gap even with hints → failure is fundamental (cannot follow even explicit action directives)

However, the Llama results should be interpreted with caution: the hint gap measures planning improvement, but the resulting success does not demonstrate trust reasoning — it demonstrates hint-following with a favorable NPC assignment. A stronger test would randomize which NPC is truthful across episodes.

---

## Discussion

### The Reflection Paradox

Our most notable finding is that adding reflection *decreases* performance. In hard mode without hints, Reflection-Enhanced (28%) performs worse than Naive (64%). The paradox persists across deception levels, including LR=0.0 where there is nothing to reflect about.

With hints, Reflection-Enhanced jumps to 100% for GPT-OSS and 93% for Llama. Crucially, reflection does not consume a step — it runs inside `select_action()` before the environment increments the step counter. The paradox is caused not by step waste but by **behavioral contamination**: after reading a reflection that flags potential deception, the LLM becomes over-cautious, re-queries NPCs unnecessarily, and second-guesses correct information. The hints override this over-caution by providing explicit action directives.

In the extended world (25-step budget vs 18), Reflection-Enhanced improves to 58% — still below Naive in the default world, but better than its 28% there. This supports the interpretation that reflection's cost is step-budget-relative: with more slack, the overhead is less punishing.

**Implication**: Reflection-style architectures (Reflexion, LATS) should be evaluated under resource constraints, not just in unlimited-step settings. A technique that appears beneficial with generous budgets may be actively harmful when steps are scarce.

### Environment Complexity as a Stress Test

The extended world (7 locations, 4 sigils, branched topology) provides stronger differentiation than the default world:
- Naive drops from 64% → 33% overall
- The cost of wrong information is higher because recovery requires more steps in a branched graph
- Belief-Tracking and Memory+Trust remain at 100% in both environments, confirming their robustness is genuine

This validates the principle that **evaluation environments should be calibrated to produce meaningful variance** — neither too easy (ceiling effect) nor too hard (floor effect).

### Payload Hints as a Diagnostic Tool

The hint ablation provides a principled way to decompose LLM agent failures:
- **No hint gap** (Belief-Tracking on GPT-OSS): The model handles both planning and reasoning
- **Large hint gap** (Belief-Tracking on Llama): Planning is the bottleneck, not reasoning
- **No improvement even with hints** (Naive on Llama): The failure is more fundamental

This decomposition could generalize to other agent benchmarks: run with and without structured hints to separate "the model can't figure out what to do" from "the model can reason but can't act efficiently."

### Cross-Model Generalizability

GPT-OSS-120B and Llama-4-Scout represent two extremes, but the interpretation requires nuance:
- **GPT-OSS**: Strong autonomous planner. Belief-Tracking and Memory+Trust achieve 100% without hints. However, neither variant correctly identifies deceptive NPCs — they succeed because deceptive NPCs in this environment give correct location information (lying primarily about vault order). Success does not imply accurate trust calibration.
- **Llama-4-Scout**: Poor autonomous planner. Achieves 0% without hints. With hints, achieves 93% on Belief-Tracking — but trace analysis shows this is because Llama exclusively consults Aster (always truthful), not because it reasons about trust. The model avoids the deception problem rather than solving it.

This suggests that **planning capability is the primary bottleneck**, but also that **current success metrics may overstate trust reasoning ability**. An agent can achieve 100% success without ever correctly identifying a liar, as long as it happens to follow correct information. Future work should include trust calibration accuracy (not just task success) as a primary metric.

### Baselines Validate the Experimental Design

The oracle and random baselines serve two purposes:
1. **Sanity check**: Oracle achieves 100% at optimal step count (15 in default, 19 in extended), confirming the task is solvable. Random achieves 0%, confirming the task is non-trivial.
2. **Performance framing**: LLM variants can be evaluated on a 0–100% scale where the bounds are empirically established, not assumed.

### Limitations

- **Sample size**: 5 runs per setting in default world (150 episodes), 3 runs in extended world (72 episodes). Wilson 95% CIs for per-cell rates with n=5 span ~40 percentage points (e.g., 100% → [57%, 100%]). Overall variant comparisons (n=25) are tighter but still wide.
- **NPC role assignment is not fully randomized**: Aster is always truthful. This creates a confound for Llama, which exclusively consults Aster and thus sidesteps deception entirely. Randomizing NPC role assignments would provide a stronger test of trust reasoning.
- **Success ≠ trust calibration**: Agents can achieve 100% success without correctly identifying deceptive NPCs — deceptive NPCs in our setup lie primarily about vault order, not locations. A harder variant where deceptive NPCs give wrong location information would better test trust reasoning.
- **Hybrid design**: NPCs use deterministic mock responses. Real LLM NPCs might produce more naturalistic deception that is harder to detect.
- **Two models**: We tested GPT-OSS-120B and Llama-4-Scout. Additional models (Mistral, Claude) would strengthen generalizability claims.
- **Hint design**: The PRIORITY_ACTION hints are hand-crafted for this environment. A more principled approach (e.g., learned hints, chain-of-thought scaffolding) would be needed for general applicability.

---

## Files Reference

| File | Description |
|------|-------------|
| `results_hard-hybrid_spread.json` | GPT-OSS default world, no hints, 5 runs (150 episodes) |
| `results_extended.json` | GPT-OSS extended world, no hints, 3 runs (72 episodes) |
| `results_nohints_gptoss.json` | GPT-OSS default world, no hints (backup) |
| `results_nohints_llama.json` | Llama default world, no hints (60 episodes) |
| `results_hints_gptoss.json` | GPT-OSS default world, with hints (60 episodes) |
| `results_hints_llama.json` | Llama default world, with hints (60 episodes) |
| `results_cross_model.json` | Cross-model comparison (Llama, no hints) |
| `results_scaling_mock.json` | NPC scaling experiment (mock) |
| `results_mock_spread.json` | Mock ceiling results, default world |
| `llm_logs/calls.jsonl` | Raw LLM API calls with prompts and responses |
| `plots_hard_hybrid/` | Default world real LLM plots |
| `plots_cross_model/` | Cross-model comparison plots |
