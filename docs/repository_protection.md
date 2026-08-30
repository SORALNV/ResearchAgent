# Main branch protection

The repository contains an idempotent ruleset definition at:

```text
.github/rulesets/main.json
```

It protects `refs/heads/main` by:

- rejecting branch deletion
- rejecting non-fast-forward updates
- requiring linear history
- requiring changes through pull requests
- allowing only squash merges
- requiring review-thread resolution
- requiring both `pytest (3.11)` and `pytest (3.12)` checks to pass on the latest base branch

The pull-request rule uses zero required approvals because this is currently a single-maintainer personal repository; requiring one approval would prevent the PR author from merging their own work. CODEOWNERS is still recorded for ownership and future multi-maintainer use.

## Activation

Creating or updating repository rulesets requires repository Administration write permission. The normal Actions token and the ChatGPT GitHub connector used for code changes do not expose that administrative mutation.

Create a fine-grained token with Administration: write for this repository, store it as the Actions secret `REPOSITORY_ADMIN_TOKEN`, and manually run:

```text
Actions -> apply-main-ruleset -> Run workflow
```

The workflow invokes:

```bash
python scripts/apply_repository_ruleset.py
```

The script is idempotent: it updates the ruleset named `Protect main` if it exists, otherwise it creates it. It does not print the token.

The same script can be run locally:

```bash
export GITHUB_ADMIN_TOKEN=...
export GITHUB_REPOSITORY=SORALNV/ResearchAgent
python scripts/apply_repository_ruleset.py
```

After activation, confirm that the repository rules page shows `Protect main` as active and that direct pushes to `main` are rejected.
