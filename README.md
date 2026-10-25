# Auditing Anachronistic Semantic Projection in Foundation Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-pep8-green.svg)](https://www.python.org/dev/peps/pep-0008/)

Evaluation harness and benchmark for auditing **temporal and anachronistic projection bias** in autoregressive foundation models.

Modern Large Language Models (LLMs) are predominantly optimized on 21st-century internet corpora. When processing historical common-law statutes or contractual instruments, these models project contemporary colloquial semantic neighborhoods onto legally fossilized tokens, generating unflagged interpretative errors.

This harness operationalizes the noise-calibrated diachronic measurement framework introduced in [Sajjad & Saleem (2026)](https://doi.org/10.5281/zenodo.23613023) to score model vulnerability against historical ground truth.

---

## Benchmark Design

Each benchmark entry pairs an enacted statutory clause with:
1. Target polysemic token $w$
2. Verified historical semantic centroid $\vec{\mu}_{\text{hist}}$
3. Contemporary colloquial distractor centroid $\vec{\mu}_{\text{mod}}$

We evaluate:
* **Conditional Surprisal:** $S(w \mid c) = -\log_2 P(w \mid c)$
* **Relative Projection Alignment:** $\Delta = \cos(\vec{h}(c), \vec{\mu}_{\text{mod}}) - \cos(\vec{h}(c), \vec{\mu}_{\text{hist}})$
* **Anachronistic Error Rate (AER):** Percentage of test cases where modern colloquial bias exceeds historical alignment ($\Delta > 0$).

---

## Quickstart

### Setup
```bash
git clone [https://github.com/Haidriyam/llm-anachronism-audit.git](https://github.com/Haidriyam/llm-anachronism-audit.git)
cd llm-anachronism-audit

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
