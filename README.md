# Query–Interface Language Mismatch in Multilingual GUI Reasoning — follow-up experiments

This repository collects the experiment and analysis code I (Gihyun Bae, [@gongdoriB](https://github.com/gongdoriB)) wrote for the follow-up stage of a two-person CUAI (Chung-Ang University AI Society) research project.

**Base project:** the model-running code, data preparation, the summer-study pipeline and released results live in my teammate's repository, [dltmdwoOS/mpr-gui-crosslocale](https://github.com/dltmdwoOS/mpr-gui-crosslocale), which we both commit to. My commits there are listed [here](https://github.com/dltmdwoOS/mpr-gui-crosslocale/commits/method-selective-alignment-dev?author=gongdoriB). This repository holds the add-ons, runners, patches and analysis scripts that sit on top of it. Most scripts expect the base repository to be checked out next to them.

## Research question

When a user asks about a GUI in one language and the screen shows text in another, how much does a vision-language model's understanding suffer — and where does the loss come from?

- **Summer study (2026, CUAI Summer Conference, Grand Prize).** A fully crossed 6 × 6 query-language × GUI-language design over 2,195 MPR-GUI-Bench items with two VLMs, analysed with item-level GLMMs. The mismatch penalty concentrated in relational questions that require locating on-screen text; a two-stage localization method recovered part of it.
- **Follow-up (Sep 2026 – present, this repository).** Separates *what* changes in the query: the referent names (swapped to the on-screen text) versus the sentence language, and compares both with machine-translated queries. Six languages and seven VLMs (Qwen3-VL, InternVL2.5/3.5, UI-TARS-1.5, MiniCPM-V 4.5 and smaller sizes). Results are preliminary and unpublished, so they are not included here; a manuscript is in preparation.

## What is here

| Folder | Contents |
| --- | --- |
| `experiments/rstem_ropts/` | Name-swap ablation that changes names only in the question stem (R_stem) or only in the answer options (R_opts); Qwen3-VL and InternVL3.5 runners and pre-registered analysis plans |
| `experiments/expA_translation/` | Experiment A: machine-translated queries (T) vs. on-screen-name queries (R) across six languages |
| `experiments/six_lang/` | Building the six-language relational-question set |
| `experiments/model_expansion/<date>/` | Add-ons, queues and runbooks used to extend the experiments to new models and languages on single-RTX-4090 servers, kept by run date |
| `experiments/internvl35_single4090/` | InternVL3.5-8B single-GPU run and grounding audit |
| `experiments/method_selective_alignment/` | Method development (Nov 2026, in progress): OCR the screen, then edit the query so its names match the on-screen text |
| `analysis/` | Item-level bootstrap CIs, Holm correction, GLMM preparation and fitting (R), per model and per experiment |
| `scripts/night_runs/` | Unattended run definitions |

Runbooks (`RUNBOOK_KO.md`) and analysis plans (`ANALYSIS_PLAN*_KO.md`) are written in Korean. Analysis rules were fixed before results were inspected.

## Notes

- Local paths were replaced with `${PROJECT_ROOT}`, `${CUAI_ROOT}` and `${HOME}`; `/workspace/...` paths refer to the remote GPU server.
- Not included: model weights, images, raw predictions, result tables and logs.
- Data derived from MPR-GUI-Bench follows the original dataset's CC BY-NC 4.0 terms. No license has been chosen for the code yet.
- Parts of the tooling were drafted with AI coding assistants; the experimental design, analysis rules and interpretation are ours.

Contact: bgh0325@cau.ac.kr
