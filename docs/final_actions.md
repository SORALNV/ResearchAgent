# Final actions: Kaggle submission and paper generation

## Responsibility boundary

ResearchAgent treats the following as the only human-owned research-direction
judgments.

| Domain | Human judgment 1 | Human judgment 2 | Human judgment 3 |
|---|---|---|---|
| Research | choose the hypothesis after Discord consultation | accept/reject/defer the interpretation of an exact result | decide whether that exact result/evidence bundle should become a paper |
| Kaggle | choose the hypothesis after Discord consultation | accept/reject/defer the interpretation of an exact result | approve the exact SHA-256 submission candidate |

After the applicable final judgment is accepted, Core performs the remaining
mechanical work. Operational safety boundaries remain separate: account setup,
competition-rule acknowledgment, credential management, paid compute,
destructive operations, and external publication are not silently automated.

## Durable final-action coordinator

The routed Discord service owns a restart-recoverable `FinalActionCoordinator`.
It scans immutable human decision events and writes one deterministic action
record per `(WorkSession, action kind, subject_ref)` under
`FINAL_ACTION_RUNTIME_DIR`.

```text
human decision event
       |
       +-- kaggle_submission accept
       |        -> candidate discovery/validation
       |        -> exact SHA gate re-check
       |        -> Kaggle CLI submit
       |        -> submission-history reconciliation
       |        -> public/private score history event
       |
       +-- research_paper accept
                -> accepted interpretation/evidence bundle
                -> literature records
                -> draft
                -> deterministic + provider review
                -> bounded revision
                -> Markdown, LaTeX, BibTeX, manifest and optional PDF
```

A later `reject` or `defer` cancels a pending final action. A successful action
is idempotent. Running actions are reconciled after a Core restart.

## Kaggle submission pipeline

### Candidate contract

An experiment result can declare a candidate in `result.json`:

```json
{
  "competition_slug": "owner-or-competition-slug",
  "submission_candidate": {
    "path": "submission.csv",
    "message": "feature set H17, locked CV 0.8123",
    "expected_columns": ["id", "target"],
    "sample_submission_path": "sample_submission.csv",
    "validation": {"valid": true},
    "risks": ["public-LB variance"]
  }
}
```

The candidate file must stay under the collected artifact root. Core verifies
that it is a non-symlink regular file, parses its CSV structure, verifies header
and row consistency, optionally compares it with `sample_submission.csv`, and
computes its SHA-256. A candidate with missing competition metadata or failed
validation is not submittable.

### Human command

Use the exact digest shown by `/agent status`:

```text
/agent submit sha256:<64 lowercase hex> verdict:accept note:<final judgment>
```

Immediately before submission Core:

1. reloads the exact candidate;
2. confirms that the latest human verdict still accepts the exact SHA-256;
3. re-hashes the file and rejects any byte change;
4. confirms that the competition is in `KAGGLE_RULES_ACKNOWLEDGED`;
5. checks existing Kaggle submission history for the deterministic marker;
6. invokes the official command:

```text
kaggle competitions submit -c <competition> -f <csv> -m <message-with-marker>
```

7. polls `kaggle competitions submissions -c <competition> -v` and records the
   submission status and available public/private scores.

If Core crashes or the network response becomes ambiguous after the submit
request, it does **not** blindly send the same file again. It first reconciles
the marker against Kaggle history. Until that succeeds, the action remains
`blocked/uncertain`.

### Configuration

```env
KAGGLE_SUBMISSION_ENABLED=true
FINAL_ACTION_RUNTIME_DIR=final_actions
FINAL_ACTION_SCAN_SECONDS=10
FINAL_ACTION_RETRY_SECONDS=30
FINAL_ACTION_MAX_FAILURE_ATTEMPTS=3
FINAL_ACTION_MAX_CONCURRENT=2

# Required before automatic submission for each competition.
# This records that account/rules setup has been completed; it is not inferred.
KAGGLE_RULES_ACKNOWLEDGED=competition-slug-1,competition-slug-2
KAGGLE_DEFAULT_COMPETITION=
KAGGLE_SUBMISSION_MAX_BYTES=536870912
KAGGLE_SUBMISSION_HISTORY_POLL_SECONDS=5
KAGGLE_SUBMISSION_HISTORY_TIMEOUT_SECONDS=90

# The official Kaggle CLI and credentials remain in Core only.
KAGGLE_COMMAND=kaggle
KAGGLE_COMMAND_TIMEOUT_SECONDS=180
KAGGLE_USERNAME=
KAGGLE_KEY=
KAGGLE_API_TOKEN=
```

`DISCORD_*` and `OPENAI_*` secrets are excluded from the Kaggle subprocess
environment. Kaggle credentials are never forwarded to experiment workers or
model subprocesses.

## Paper-generation pipeline

### Human command

The normal target is an exact accepted result reference:

```text
/agent paper result_ref:result:<job-id>:<hash> verdict:accept note:<paper judgment>
```

Paper generation requires both:

- an accepted `research_paper` decision for the subject; and
- an accepted human `result_interpretation` for every result included in the
  evidence bundle.

### Generated artifacts

Each deterministic paper ID receives a directory containing:

```text
paper_outputs/<work-session>/<paper-id>/
├── evidence.json
├── draft_v001.md
├── draft_v002.md             # when revision is required
├── paper.md
├── paper.tex
├── references.bib
├── review.json
├── paper_manifest.json
├── pipeline_state.json
└── paper.pdf                  # only when optional LaTeX compilation succeeds
```

The evidence bundle contains the durable Project, WorkSession, JobSpecs,
accepted result events, artifact references, human interpretations, and
retrieved literature records. The writer is instructed to use only this bundle.
Citation syntax is restricted to known keys such as `[@P001]`; unknown keys are
removed by deterministic validation. Required sections are checked and repaired
without inventing missing evidence.

The pipeline performs a bounded writer/reviewer/revision loop through the
existing provider policy. When no provider is configured or a provider fails,
it emits a conservative evidence-derived manuscript rather than fabricating a
successful provider call.

External publication, conference submission, and authorship assertions are not
performed by this pipeline.

### Configuration

```env
PAPER_PIPELINE_ENABLED=true
PAPER_OUTPUT_DIR=paper_outputs
PAPER_MAX_SOURCES=10
PAPER_MAX_REVISIONS=2
PAPER_SEARCH_TIMEOUT_SECONDS=15
PAPER_PROVIDER=arxiv
PAPER_ARTIFACT_MAX_FILES=500
PAPER_ARTIFACT_MAX_BYTES=268435456

# Markdown/LaTeX/BibTeX are always produced. PDF compilation is optional.
PAPER_COMPILE_PDF=false
PAPER_LATEX_COMMAND=latexmk
PAPER_COMPILE_TIMEOUT_SECONDS=180
```

## Status and recovery

`/agent status` now includes:

- compute jobs and result references;
- pending hypothesis proposals;
- Kaggle candidates, submission state and available LB scores; or
- paper artifact paths and generation state;
- durable final-action records and retry state.

On startup the service:

1. recovers compute jobs;
2. reloads nonterminal final actions;
3. scans existing accepted final decisions;
4. discovers unregistered submission candidates;
5. reconciles ambiguous Kaggle submissions against history; and
6. resumes blocked actions when their dependencies become available.

## Validation boundary

CI uses an injected Kaggle transport and deterministic paper writer/search
fixtures. It exercises command construction, SHA revalidation, duplicate
suppression, history/LB parsing, evidence collection, citation constraints,
artifact generation, and restart recovery without using live credentials.
A live submission occurs only in a deployed Core where a real candidate,
competition-rule acknowledgment, Kaggle credentials, and an exact human SHA
approval all exist.
