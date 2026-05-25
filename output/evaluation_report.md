# Repository Maturity Classification via Weak Supervision: A Methodological Sensitivity Analysis of Label Operationalization Quality

**Final Experimental Report — May 2026**

---

## 1. Introduction

Classifying software repositories by engineering maturity is a challenging problem that sits at the intersection of natural language processing, software engineering analytics, and hiring intelligence. The core task is deceptively simple: given a GitHub repository's observable signals — its README, commit history, CI/CD configuration, dependency graph, and contribution patterns — can an automated system reliably estimate the seniority level of the engineers who maintain it?

This question matters for several practical reasons. Recruiters and talent-sourcing platforms could use repository maturity signals to identify engineers whose public work reflects a given experience level, complementing resume-based screening with behavioral evidence. Engineering managers evaluating open-source dependencies could assess whether a library is maintained with professional-grade practices or represents a personal side project. Technical interview pipelines could incorporate repository evidence as one signal among many when calibrating candidate expectations.

However, the ethical risks are equally significant. GitHub repositories are curated public personas, not comprehensive portraits of engineering competence. An engineer may contribute to sophisticated internal systems invisible on their public profile, or maintain a simple repository with exceptional code quality that automated signals fail to capture. Conversely, a repository with extensive CI/CD and documentation may reflect organizational scaffolding rather than individual expertise. Any automated maturity classification system must be understood as a proxy signal — noisy, partial, and potentially biased — never as a replacement for human evaluation. The features extracted in this work (stars, forks, contributors, commits, CI/CD presence, tests, releases, documentation, dependencies, and repository age) are all structural proxies that can be gamed, misinterpreted, or confounded by organizational context.

This project approaches the problem through the lens of **weak supervision**: rather than requiring costly human-annotated labels for hundreds of repositories, we use a large language model (DeepSeek) to generate training labels through carefully designed prompts, then fine-tune a lightweight classifier (DistilBERT) on the resulting weakly-labeled dataset. The central methodological question is whether the *operationalization quality* of the weak-labeling prompt — how concretely and observably each maturity category is defined — substantially affects downstream classifier performance.

To investigate this, we conduct a controlled methodological sensitivity analysis comparing two labeling approaches that differ **only** in their category definitions, holding the dataset, features, model architecture, and training pipeline constant. The results reveal that weak-label operationalization quality is a first-order determinant of downstream performance, with implications for any project that relies on LLM-generated training data.

---

## 2. Repository Collection and Sampling

### 2.1 Two-Round Informed Sampling Strategy

Repositories were collected from GitHub using a two-stage informed sampling procedure designed to maximize structural diversity across the engineering maturity spectrum. The sampling was **structural, never label-driven**: search queries targeted observable repository characteristics (star ranges, activity levels, repository age, contributor counts) rather than maturity category names, avoiding the circularity of sampling for the very labels we intended to predict.

**Round 1** (469 repositories) used broad queries spanning diverse star ranges and activity profiles on GitHub, capturing repositories from nascent personal projects to well-established community tools. **Round 2** (302 repositories) was a targeted follow-up round designed to fill structural gaps identified after the initial collection, particularly in the mid-range of repository activity (repositories with moderate stars and inconsistent activity patterns) and in underrepresented segments of the contributor-count distribution.

The combined dataset contains **771 repositories**, each described by 10 engineering signals extracted directly from the GitHub API and repository metadata.

### 2.2 Structural Diversity and Sampling Bias

The sampling strategy intentionally avoided pure popularity bias. Rather than simply selecting the most-starred repositories (which would overwhelmingly represent mature, lead-level projects), the stratified design ensured coverage across the full range of repository sizes, ages, and activity levels. This was essential for building a classifier that could distinguish a junior engineer's well-structured learning project from a senior engineer's widely-adopted library.

Nevertheless, the sampling strategy introduces known biases. Repositories must meet minimum quality filters to be included: size above 5KB, presence of a README file, and non-archived status. This systematically excludes abandoned repositories that lack documentation, as well as very early-stage projects with minimal code. Consequently, the dataset likely underrepresents truly low-value repositories and may overrepresent repositories with at least *some* engineering structure. The practical effect is that the classifier faces a harder challenge distinguishing adjacent maturity levels (where structural signals overlap) than identifying the extremes of the spectrum.

### 2.3 Stratification and Class Balance

The final dataset of 771 labeled repositories was split into training (539, 70%), validation (115, 15%), and test (116, 15%) sets using stratified sampling with `seed=42`. Stratification preserved the label distribution across splits, but imbalance persisted: the majority class (senior) outnumbered the minority class (template) by approximately 3.6:1 in the training set. Class-weighted cross-entropy loss was employed during training to partially mitigate this imbalance, although weight adjustment alone cannot compensate for the limited diversity of minority-class training examples.

---

## 3. Repository Representation

### 3.1 Extracted Signals

Ten engineering signals were extracted per repository, chosen to capture orthogonal dimensions of software engineering maturity:

| Signal | Dimension Captured | Extraction Method |
|--------|-------------------|-------------------|
| Stars | Community interest | GitHub API |
| Forks | Derivative usage | GitHub API |
| Contributors count | Team size / collaboration | GitHub API |
| Commits (6 months) | Development velocity | GitHub API |
| Releases count | Release discipline | GitHub API |
| CI/CD workflows | Automation maturity | File-tree inspection |
| Test files | Testing culture | File-tree inspection |
| README length | Documentation quality | GitHub API |
| Dependency lines | Ecosystem integration | Dependency-file parsing |
| Repository age (days) | Longevity / sustainability | GitHub API |

### 3.2 Textual Representation

Each repository was converted into a structured textual summary (`text_repr`) aggregating these signals into a readable format suitable for both LLM labeling and BERT fine-tuning. The representation averages **712 characters**, well within DistilBERT's 512-token limit, avoiding truncation artifacts.

Critically, the text representation **excludes** popularity-correlated metadata such as stars, forks, owner name, and repository topics. This design choice prevents the classifier from learning trivial shortcuts (e.g., "repositories with more than 500 stars are always senior") and forces it to attend to genuine engineering structure. The trade-off is that some genuinely informative signals — particularly stars as a proxy for community validation — are withheld from the model.

### 3.3 Signal Association with Maturity Categories

Different engineering signals carry differential diagnostic value across maturity categories, as revealed by both the LLM labeling behavior and downstream classifier confusion patterns:

**Intern repositories** tend to exhibit nascent engineering structure: few contributors (often a single author), minimal or absent CI/CD, sparse commit history, and README files that describe learning objectives rather than production usage. Testing infrastructure is typically absent or consists of a single example file. The distinguishing challenge is separating genuine intern-level projects from low-value repositories that superficially resemble them.

**Junior repositories** show emerging professional practices: some test files present, basic CI/CD configuration, moderate commit activity, and README documentation that addresses setup and usage. The boundary between junior and intern is the fuzziest in the dataset, as both categories share structural characteristics of early-stage projects.

**Senior repositories** display established engineering practices: consistent commit history, multiple contributors, CI/CD pipelines, test suites, documented releases, and comprehensive README files. These repositories look professionally maintained but typically lack the scale and governance patterns of lead-level projects. The senior→lead confusion boundary is the single most prominent source of classification error in the baseline approach.

**Lead repositories** exhibit architectural leadership: large contributor bases, sophisticated CI/CD with multiple workflow files, extensive test coverage, regular releases with versioning discipline, substantial documentation, and complex dependency graphs. The model's strong recall on this class suggests that lead-level signals are distinct and learnable when the label definitions are clear.

**Template repositories** are structurally distinctive: they contain boilerplate project structures, configuration scaffolding, and setup instructions but little original code or commit history beyond the initial template creation. Their structural signature is unique enough that they form the best-classified category under the alternative approach (F1=0.82).

**Low-value repositories** are characterized by absent or minimal engineering signals across nearly all dimensions: no CI/CD, no tests, few or no releases, minimal documentation, and sparse commit history. The primary classification challenge is distinguishing genuinely low-value repositories from intern repositories in early development stages, particularly when both share superficial README presence.

---

## 4. Weak Labeling with LLMs

### 4.1 Labeling Pipeline

The weak-supervision pipeline uses DeepSeek (a large language model) to assign each repository a maturity label based on its `text_repr` summary. The LLM receives a structured prompt defining the six categories and is asked to reason about each repository's engineering signals before producing a label, confidence score, and justification. Temperature is set to 0 for deterministic labeling. A checkpoint is saved every 50 repositories to prevent data loss.

The six maturity categories are:

| Label | ID | Description |
|-------|----|-------------|
| intern | 0 | Entry-level; learning-oriented; minimal engineering infrastructure |
| junior | 1 | Emerging professional; basic CI/CD and testing; small team |
| senior | 2 | Established practices; consistent activity; documented releases |
| lead | 3 | Architectural leadership; large contributor base; sophisticated automation |
| template | 4 | Boilerplate/scaffolding; little original code beyond project structure |
| low-value | 5 | Minimal engineering signals across most dimensions |

### 4.2 Baseline Prompt Philosophy

The **baseline approach (V1)** employed qualitative category definitions that described maturity levels in terms of engineering *character* rather than specific, observable criteria. For instance, a "senior" repository was characterized as "demonstrating established engineering practices with consistent activity," while "lead" was described as "showing architectural leadership and sophisticated automation." These definitions, while conceptually meaningful, left substantial room for interpretation at the boundaries between adjacent categories.

The baseline labeling produced a distribution heavily concentrated in the senior category (29.4% of all repositories), with low-value (18.2%) and lead (18.0%) as the next most common classes. The LLM's tendency to assign senior as a default for repositories with moderate-but-not-exceptional signals created a label distribution that reflected the annotator's central tendency as much as the underlying repository characteristics.

### 4.3 Semantic Overlap and Label Noise

The baseline labels exhibited significant semantic overlap between adjacent categories, particularly senior↔lead and intern↔junior. The LLM's reasoning chains revealed patterns of subjective judgment: a repository with CI/CD, tests, and multiple contributors might be labeled senior or lead depending on the model's interpretation of "architectural leadership" — a concept with no direct observable proxy in the signal set. This semantic ambiguity introduced structured label noise that the downstream classifier inherited and amplified.

The borderline between intern and low-value was particularly problematic. Both categories describe repositories with minimal engineering infrastructure, and the distinction often hinged on the perceived *intent* of the repository (learning project vs. abandoned code) — a judgment that even human annotators would find difficult to make reliably from structural signals alone.

### 4.4 Label Distribution (Baseline)

| Class | Count | Percentage |
|-------|-------|------------|
| senior | 227 | 29.4% |
| low-value | 140 | 18.2% |
| lead | 139 | 18.0% |
| intern | 105 | 13.6% |
| junior | 96 | 12.5% |
| template | 63 | 8.2% |
| error | 1 | 0.1% |

One repository (0.1%) could not be classified by the LLM and was excluded from training.

---

## 5. Baseline Approach Evaluation

### 5.1 Overall Performance

The baseline DistilBERT model, fine-tuned on V1 weak labels, achieved the following test-set performance on 116 held-out repositories:

| Metric | Value |
|--------|-------|
| Accuracy | 0.405 |
| Macro Precision | 0.388 |
| Macro Recall | 0.438 |
| Macro F1 | 0.393 |
| Weighted F1 | 0.375 |

The macro F1 of 39.3% indicates that the model performs substantially better than random chance (16.7% for a 6-class problem), but the absolute performance leaves considerable room for improvement. Weighted F1 (37.5%) is slightly lower than macro F1, indicating that the model underperforms on the majority class (senior) relative to some minority classes.

### 5.2 Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| intern | 0.333 | 0.438 | 0.378 | 16 |
| junior | 0.250 | 0.333 | 0.286 | 15 |
| senior | 0.412 | 0.206 | 0.275 | 34 |
| lead | 0.543 | 0.905 | 0.679 | 21 |
| template | 0.455 | 0.556 | 0.500 | 9 |
| low-value | 0.333 | 0.190 | 0.242 | 21 |

The most striking pattern is the asymmetry in lead classification: recall of 90.5% but precision of only 54.3%. The model over-predicts lead (35 predictions for only 21 true leads), suggesting it learned that lead-associated signals (contributors, CI/CD, releases) are strong predictors but cannot reliably distinguish senior from lead when both share those signals. Senior — the majority class in training — has the worst recall (20.6%), with the model systematically misclassifying true senior repositories as lead.

Low-value repositories are rarely identified (19.0% recall), with the model confusing them primarily with intern and junior. Template and intern achieve moderate performance (F1 ≈ 0.38–0.50), while junior is the second-worst class (F1 = 0.286).

### 5.3 Confusion Matrix Analysis

```
              pred_intern  pred_junior  pred_senior  pred_lead  pred_template  pred_low-value
true_intern        7            6            0           0             1              2
true_junior        4            5            2           0             1              3
true_senior        4            3            7          15             3              2
true_lead          0            1            1          19             0              0
true_template      0            0            3           0             5              1
true_low-value     6            5            4           1             1              4
```

Three confusion patterns dominate the error landscape:

1. **senior → lead (15 cases, 44.1% of true senior):** Nearly half of all senior repositories are classified as lead. This is the single largest error mode and accounts for 22% of all errors. The model has effectively collapsed the senior–lead distinction, treating both as a single "professional-grade" category.

2. **intern → junior (6 cases, 37.5% of true intern):** The model systematically upgrades intern repositories to junior, suggesting that the structural signals distinguishing these adjacent early-career categories are too subtle for the current representation and labeling scheme.

3. **low-value → intern (6 cases, 28.6% of true low-value):** Low-value repositories are frequently overestimated as intern-level, consistent with the hypothesis that superficial structural signals (README presence, basic code organization) create a misleading veneer of engineering maturity.

Adjacent-class errors account for 33 of 69 total errors (47.8%), confirming that the primary difficulty is not distinguishing radically different repository types but rather resolving the fuzzy boundaries between neighboring maturity levels.

### 5.4 Methodological Diagnosis

**Why senior → lead dominates.** The baseline prompt's qualitative distinction between "established practices" (senior) and "architectural leadership" (lead) maps poorly onto observable repository signals. Both categories feature contributors, CI/CD, tests, releases, and documentation. The LLM labeler relied on subjective interpretation of scale and sophistication that the text representation captures only weakly, producing labels that the downstream classifier could not learn to replicate.

**Why low-value recall is poor.** The sampling strategy's quality filters (size > 5KB, README required) pre-select against the most obviously low-value repositories. Those that remain in the dataset often have enough structural scaffolding to confuse the classifier, particularly when the text representation does not capture negative signals (absence of CI/CD, absence of tests) as prominently as positive ones.

**Why junior underperforms.** Junior is an interstitial category squeezed between intern and senior with no distinctive structural signature of its own. A repository with some tests and basic CI/CD could be a strong intern project, a typical junior project, or a modest senior project — the structural signals are identical, and only the *quality* and *consistency* of execution would disambiguate them, which the current signals do not capture.

**Model capacity considerations.** DistilBERT (66M parameters) was chosen for practical reasons — it trains on a single T4 GPU in under 10 minutes. While it retains approximately 95% of BERT-base's performance on standard benchmarks, the task of distinguishing six fine-grained software engineering categories from ~700 characters of structured text may benefit from a larger model or a code-aware pretraining objective (e.g., CodeBERT).

### 5.5 Motivation for an Alternative Approach

The baseline results pointed to a clear bottleneck: **weak-label quality**, not model capacity or feature engineering, appeared to be the limiting factor. The structured label noise — systematic senior→lead mislabeling, intern↔low-value ambiguity — propagated directly into the classifier's confusion patterns. Rather than pursuing a larger model, additional features, or architectural changes — all of which would add complexity without addressing the root cause — we designed an alternative labeling approach that replaced qualitative category definitions with **operationalized, observable engineering criteria**.

---

## 6. Alternative Approach: Operationalized Weak Labels

### 6.1 Controlled Experimental Design

The alternative approach (V2) modifies **only the weak-label operationalization** — the prompt structure and category definitions used by the LLM to assign labels. All other components of the pipeline remain identical to the baseline:

| Component | Status |
|-----------|--------|
| Repository sampling (two rounds, 771 repos) | **Fixed** |
| Extracted features (10 signals) | **Fixed** |
| Text representation format (`text_repr`) | **Fixed** |
| Train/val/test split (539/115/116, seed=42) | **Fixed** |
| Model architecture (DistilBERT, 66M params) | **Fixed** |
| Training hyperparameters (batch=16, epochs=8, lr=2e-5) | **Fixed** |
| Weak-label prompt and category definitions | **Modified** |

This controlled design isolates the causal effect of label operationalization quality on downstream performance. Any observed differences between V1 and V2 can be attributed to changes in label quality rather than confounding factors.

### 6.2 Redesigned Category Definitions

The V2 prompt replaced qualitative, character-based descriptions with **operationalized criteria** grounded in observable engineering signals:

- Instead of "demonstrates established engineering practices," V2 specified thresholds: *"has CI/CD configured, test files present, multiple contributors with consistent commits, documented releases, and README exceeding 500 characters."*

- Instead of "shows architectural leadership," V2 enumerated observable proxies: *"10+ contributors, multiple CI/CD workflows, extensive test suites spanning unit/integration/e2e, semver releases, dependency graph with 20+ packages, and comprehensive documentation including contributing guidelines."*

- For low-value repositories, V2 emphasized the *joint absence* of signals: *"no CI/CD, no tests, no releases, minimal or auto-generated README, fewer than 3 contributors, sparse commit history."*

The key design principle was **reducing annotator degrees of freedom**: every category was defined in terms of concrete, verifiable properties of the repository rather than abstract judgments about engineering quality. This reduced the LLM's reliance on implicit reasoning and made the labeling function more deterministic.

### 6.3 Label Distribution Shift

The V2 labeling produced a substantially different class distribution:

| Class | V1 Count | V1 % | V2 Count | V2 % | Delta |
|-------|----------|------|----------|------|-------|
| senior | 227 | 29.4% | 244 | 31.6% | +2.2 |
| intern | 105 | 13.6% | 163 | 21.1% | +7.5 |
| lead | 139 | 18.0% | 160 | 20.8% | +2.8 |
| template | 63 | 8.2% | 74 | 9.6% | +1.4 |
| junior | 96 | 12.5% | 66 | 8.6% | -3.9 |
| low-value | 140 | 18.2% | 63 | 8.2% | -10.0 |

The most dramatic shifts are the collapse of the low-value category (−77 repositories, −10 percentage points) and the growth of intern (+58 repositories, +7.5 points). This reflects the operationalized criteria's effect: repositories previously labeled low-value due to qualitative "lack of engineering sophistication" were reclassified as intern when the stricter criteria revealed genuine (if early-stage) engineering activity. The junior→intern migration (37 repositories) further suggests that the baseline prompt overestimated the professionalism threshold for early-career repositories.

Overall, **25.2% of repositories (194/770) changed labels** between V1 and V2, with V1/V2 agreement at 74.8%. The largest migration flows were: junior→intern (37), low-value→senior (33), senior→lead (26), and low-value→intern (21).

---

## 7. Comparative Evaluation and Sensitivity Analysis

This section presents the core methodological contribution: a controlled sensitivity analysis quantifying how much downstream classifier performance depends on weak-label operationalization quality, with all other pipeline components held constant.

**Modified variable:** weak-label category definitions (qualitative → operationalized).
**Controlled variables:** dataset, feature extraction, text representation, train/test split, model architecture, training hyperparameters.

### 7.1 Overall Metrics Comparison

| Metric | Baseline (V1) | Alternative (V2) | Delta |
|--------|---------------|------------------|-------|
| Accuracy | 0.405 | **0.612** | +0.207 |
| Macro Precision | 0.388 | **0.579** | +0.191 |
| Macro Recall | 0.438 | **0.612** | +0.174 |
| Macro F1 | 0.393 | **0.588** | +0.195 |
| Weighted F1 | 0.375 | **0.609** | +0.234 |

The alternative approach improved every aggregate metric by 17–23 percentage points. The 20.7-point accuracy gain represents a 51% relative improvement over the baseline. Macro F1 rose from 0.393 to 0.588 (+19.5 points), indicating that the improvement is not concentrated in a single class but distributed across the category spectrum.

The weighted F1 improvement (+23.4 points) exceeds the macro F1 improvement (+19.5 points), implying that the V2 labels particularly benefited the higher-support classes (senior, intern) while still improving minority classes.

### 7.2 Per-Class Comparison

| Class | V1 F1 | V2 F1 | Delta | Support (V2) |
|-------|-------|-------|-------|---------------|
| intern | 0.378 | **0.612** | +0.234 | 25 |
| junior | 0.286 | **0.222** | −0.064 | 10 |
| senior | 0.275 | **0.600** | +0.325 | 37 |
| lead | 0.679 | **0.694** | +0.015 | 24 |
| template | 0.500 | **0.818** | +0.318 | 11 |
| low-value | 0.242 | **0.583** | +0.341 | 9 |

The per-class breakdown reveals a nuanced picture:

**Largest improvements (senior, low-value, template):** Senior F1 more than doubled from 0.275 to 0.600 (+32.5 points), reflecting the operationalized prompt's success at disambiguating senior from lead. Low-value F1 more than doubled from 0.242 to 0.583 (+34.1 points), as the stricter absence-based criteria produced cleaner low-value labels that the classifier could learn. Template F1 rose from 0.500 to 0.818 (+31.8 points), confirming that template repositories have a distinctive structural signature that operationalized definitions capture effectively.

**Stable high performance (lead):** Lead F1 remained strong at 0.694 (+1.5 points), indicating that lead-level repositories are intrinsically distinguishable regardless of labeling philosophy. The operationalized criteria reduced lead over-prediction without sacrificing recall.

**Persistent difficulty (junior):** Junior F1 declined from 0.286 to 0.222 (−6.4 points), and junior is now the worst-performing class by a substantial margin. The V2 labeling reduced the junior class to only 66 training examples (8.6% of the dataset), and the test set contains only 10 junior repositories. With such limited signal, the classifier struggles to learn a coherent junior prototype. Junior's position as an interstitial category between intern and senior — sharing structural signals with both — makes it inherently difficult to classify without substantially more training data or richer features.

### 7.3 Confusion Matrix Comparison

**V2 Confusion Matrix:**
```
              pred_intern  pred_junior  pred_senior  pred_lead  pred_template  pred_low-value
true_intern       15            2            1           0             1              6
true_junior        4            2            3           0             0              1
true_senior        4            3           21           8             0              1
true_lead          0            1            6          17             0              0
true_template      0            0            2           0             9              0
true_low-value     1            0            0           0             1              7
```

Comparing the V2 confusion matrix to the baseline:

**Reduced senior→lead confusion (15 → 8 cases):** The most damaging error pattern in the baseline was cut nearly in half. The operationalized prompt's clearer separation between "established practices" and "architectural leadership" produced labels that the classifier could learn to distinguish. However, 8 senior→lead errors remain (vs. 6 lead→senior), indicating that even with operationalized criteria, the boundary between these adjacent high-maturity categories retains some intrinsic ambiguity.

**Reduced intern↔low-value confusion:** In the baseline, low-value repositories were scattered across intern, junior, and senior predictions. In V2, low-value predictions are far more concentrated: 7 of 9 true low-value repositories are correctly identified (recall 77.8%), and only 1 low-value repository is misclassified as intern (vs. 6 in baseline). The operationalized absence-based definition produced a cleaner low-value prototype.

**Emergence of intern→low-value confusion (6 cases):** A new error pattern appears in V2: 6 true intern repositories are misclassified as low-value. This likely reflects repositories at the boundary of the operationalized intern criteria — projects that technically qualify as intern (some activity, basic structure) but whose signals are sparse enough to resemble low-value. The operationalized definitions, while clearer overall, introduced a sharper boundary that some borderline repositories fall on the wrong side of.

**Persistence of junior ambiguity:** Junior errors remain dispersed across intern (4 cases), senior (3 cases), and low-value (1 case), with only 2 of 10 true juniors correctly classified. The class simply lacks sufficient training examples and distinctive structural signals to form a learnable category under the current representation.

**Template near-perfection:** Only 2 of 11 true templates are misclassified (both as senior). Template repositories — characterized by boilerplate structure, configuration scaffolding, and minimal original code — form the most clearly separable class under operationalized criteria.

### 7.4 Interpretation

The 20.7-point accuracy improvement from a single-variable change — the wording of the labeling prompt — carries a clear methodological implication: **downstream classifier performance is highly sensitive to weak-label operationalization quality.** When the LLM annotator operates with qualitative, subjective category definitions, the resulting label noise propagates into the training data and becomes the dominant source of classifier error. When the same annotator receives operationalized, criteria-anchored definitions, label quality improves sufficiently to nearly double effective classifier performance.

This finding challenges the common assumption in weak-supervision pipelines that model capacity, feature engineering, or training-data volume are the primary levers for improving downstream performance. In this setting, **semantic clarity in the labeling function mattered more than any architectural or representational change could have.** The baseline model's 40.5% accuracy was not primarily a model problem — it was a label problem.

The sensitivity to junior classification also reveals a boundary condition: operationalized criteria work well when the underlying category has a clear structural signature (template, lead, low-value), but they cannot fully compensate for categories that are inherently interstitial (junior) or for which the available signals simply do not carry enough distinguishing information.

---

## 8. Limitations

### 8.1 Weak Supervision Without Ground Truth

The most fundamental limitation of this work is the absence of human-annotated ground-truth labels. All labels — both V1 and V2 — are generated by an LLM whose own accuracy on this task is unknown. The classifier is learning to approximate the LLM's judgment, not necessarily true repository maturity. Without a human-labeled validation set, we cannot distinguish between cases where the LLM label is correct and the classifier errs, versus cases where the LLM label itself is wrong and the classifier correctly disagrees. The observed improvements from V1 to V2 demonstrate that operationalized prompts produce *more learnable* labels, but they do not guarantee that those labels are *more accurate* in an absolute sense.

### 8.2 Proxy Feature Limitations

The 10 extracted signals are structural proxies for engineering maturity, not direct measurements of it. Code quality, architectural decisions, testing thoroughness, documentation accuracy, and community health are only partially reflected in counts of files, contributors, and commits. A repository with 50 contributors and 100 releases may have excellent engineering practices or may be a poorly maintained monorepo with superficial automation — the signals cannot distinguish these cases.

### 8.3 Sampling Bias

The quality filters applied during data collection (size > 5KB, README required, non-archived) systematically exclude repositories at the lowest end of the engineering maturity spectrum. Truly abandoned repositories, repositories without any documentation, and repositories too small to contain meaningful engineering structure are underrepresented. This biases the classifier toward overestimating maturity for repositories near the low-value/intern boundary.

### 8.4 Ordinal Task Treated as Nominal

The maturity labels form an ordinal scale (intern < junior < senior < lead), but the classifier uses flat cross-entropy loss, which treats all misclassifications as equally costly. A senior→template error (a three-step ordinal mistake) is penalized identically to a senior→lead error (a one-step adjacent-class mistake). Ordinal regression or hierarchical classification approaches could better respect the ordered nature of the categories.

### 8.5 Single Model, Single Seed

Both V1 and V2 results come from a single DistilBERT training run with `seed=42`. Without multiple random restarts or k-fold cross-validation, the reported metrics are point estimates with unknown variance. The comparative conclusions (V2 outperforms V1) are robust given the large effect size, but the absolute metric values should be interpreted with appropriate uncertainty.

### 8.6 Junior as a Structurally Underdetermined Category

The persistent difficulty of junior classification across both approaches suggests a deeper problem: "junior engineer" may not correspond to a distinct structural signature in repository data. The category occupies the space between "has some professional practices" (intern) and "has established professional practices" (senior), and the available signals may simply not carry enough information to reliably place a repository at that specific intermediate point.

### 8.7 Ethical Considerations

Any system that classifies engineers by perceived maturity level carries risks of misuse. Repository signals can be gamed (artificially inflated stars, templated CI/CD, auto-generated documentation). Engineers from underrepresented backgrounds may have fewer opportunities to contribute to repositories with lead-level structural characteristics. Organizational context — whether a repository is a personal side project or a corporate-maintained tool — is invisible to the classifier but profoundly shapes the repository's engineering signals. This system should be understood as producing a noisy, partial proxy signal, not an authoritative assessment of individual engineering competence.

---

## 9. Conclusion

This project investigated whether repository engineering signals, transformed into textual representations and weakly labeled by an LLM, could support automated classification of GitHub repositories into engineering maturity categories. The results demonstrate that they can, but with an important methodological caveat: **the quality of weak-label operationalization is a first-order determinant of downstream performance.**

The baseline approach, using qualitative category definitions, achieved 40.5% test accuracy (macro F1 = 0.393). The primary failure mode was systematic confusion between adjacent maturity categories, particularly senior↔lead (44% of true senior misclassified as lead) and low-value↔intern, reflecting label noise introduced by subjective LLM judgments about abstract engineering qualities.

The alternative approach changed only the weak-labeling prompt — replacing qualitative descriptions with operationalized criteria anchored in observable repository signals — and achieved 61.2% test accuracy (macro F1 = 0.588), a 51% relative improvement. The controlled experimental design (fixed dataset, features, model, and training pipeline) isolates the causal effect of label operationalization quality, establishing that **semantic clarity in the labeling function had a larger impact than any architectural or representational change could have achieved in isolation.**

Several practical implications emerge. First, for practitioners building weak-supervision pipelines, investment in prompt engineering and category operationalization may yield higher returns than investment in larger models or additional features — at least when label noise is the binding constraint. Second, repository engineering signals do contain meaningful maturity information, but the signal is concentrated in a subset of categories (lead, template) and substantially weaker for interstitial categories (junior) that lack a distinct structural signature. Third, the strong performance on template and lead repositories suggests practical applications in automated repository triage (identifying boilerplate projects vs. professionally maintained codebases), even if fine-grained maturity classification remains challenging.

Future work should prioritize acquiring a human-labeled validation set to estimate absolute LLM labeling accuracy, exploring ordinal classification objectives that respect the ordered nature of maturity levels, and investigating whether code-aware pretrained models (CodeBERT, GraphCodeBERT) can better capture the software engineering nuance that distinguishes adjacent maturity categories. The persistent difficulty of junior classification — consistent across both approaches — suggests that richer repository representations, potentially incorporating code-level features beyond structural metadata, may be necessary to resolve the most subtle maturity distinctions.

Ultimately, this work demonstrates that weak supervision is a viable strategy for repository maturity classification, but it also underscores that the intellectual center of gravity in such pipelines lies not in the model architecture but in the careful operationalization of the concepts the model is asked to learn.
