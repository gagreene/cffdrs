# Contribution procedures

These procedures apply to changes proposed for cffdrs, by people and by coding
agents. The root [`README.md`](../README.md#development) covers the shorter setup
instructions. **Read this document before creating a branch, a commit, a pull
request, a tag or a release.**

## Branches, commits, and pushes

1. Create a focused work branch from its intended PR base under the versioned
   routing below, and check it out in the main repository folder (do not use a
   separate git worktree for a work branch). Keep unrelated edits, generated
   results, temporary files and local reference data out of the change. Check the
   staged diff before committing, and add explicit paths instead of `git add -A`.
2. **Local-only files are never committed or pushed.** This covers plans, task
   ledgers and briefs, scratch and session files, PR-description drafts, agent
   instruction files (`AGENTS.md`, `.claude/`, `.ai/`), debug symbol files
   (`*.pdb`), `rust/target/` (including the oracle baseline and benchmark output),
   `mutants-out/`, and anything under `development/` or `.superpowers/`. After
   committing, check `git diff --name-only <base> HEAD` for such paths.
3. Use commits that describe the actual change. Do not add AI authorship or
   attribution statements to commit messages, PR titles or PR descriptions (no
   `Co-Authored-By` trailers and no "generated with" lines). Commits are authored
   under the maintainer's GitHub identity, using the GitHub `noreply` address
   (`<id>+<user>@users.noreply.github.com`); a personal e-mail address is
   rejected by GitHub's push protection and must not appear in any file (author
   fields in `pyproject.toml` and source headers carry the name only). The
   maintainer reviews and is responsible for every commit. Commit titles follow
   Conventional Commits (`feat:`, `fix:`, `docs:`, `build:`, `ci:`, `test:`,
   `refactor:`, `chore:`, with an optional scope such as `cffbps` or
   `cffbps-rust`); a `!` after the type or scope marks a breaking change.
4. Coding agents must obtain the user's **direct permission** before creating or
   amending any commit. They must obtain **separate direct permission** before
   pushing a commit, branch or tag, before force-pushing, and before opening,
   editing or commenting on a pull request or review thread. Permission to edit,
   test, stage, prepare a PR, or continue work does not grant permission to
   commit or push, and permission for one action does not carry over to the next.
   Do not automate commits or pushes without explicit authorization for each
   action. After preparing and validating a change, stop, report what is ready,
   and ask separately before committing and before pushing.

## Versioned merge routing

The Python distribution version is the `version` line in
`rust/crates/cffdrs-py/Cargo.toml`. The pure Rust crate has its own version in
`rust/crates/cffbps-rust/Cargo.toml`, because applications may consume it
independently; a breaking change to its public Rust API bumps its minor version.
Version numbers use `MAJOR.MINOR.PATCH`. Route every versioned change as follows:

| Change for the next version | Work branch starts from | PR target |
| --- | --- | --- |
| Major or minor version change | `pre_release` | `pre_release` |
| Patch version change | `master` | `master` |

Patch changes go only to `master`; do not target `master` for a major or minor
change as a substitute for `pre_release`. When the major or minor release is
ready, the repository owner reviews a PR from `pre_release` into `master` and
merges it. This promotion requires a PR just like every other branch merge.
After a patch release lands on `master`, bring `master` back into `pre_release`
(through a PR) so the two do not drift. Decide which of major, minor or patch a
change is for the Python distribution when choosing the PR target; the crate
version follows its own rule above.

Release tags use the `vMAJOR.MINOR.PATCH` form (for example, `v0.2.0`), must equal
the `cffdrs-py` version above, and the release workflow checks this. The tag
`v0.1.0` exists. Tags are created on `master` after the merge, and only with the
owner's direct permission.

## Code style: function order

Every module keeps its functions in alphabetical order (case-insensitive). The rule
covers module-level functions, the methods of a class or `impl` block, and the
definitions inside a conditional block (Python `if` / `try`) or an inline Rust `mod`,
in source, tools and tests alike. Each block is ordered on its own.

- **Python:** names that start with an underscore come first (dunder methods lead,
  then the other private helpers), then the public names. A property and its setter count as one
  entry and stay together.
- **Rust:** functions without `pub` come first, then `pub` functions.
- **Order that must stay:** a function used when the module is imported (in a
  decorator, a default argument or a module constant) stays above its user. Mark it
  with `# order-exempt: <reason>` (Rust: `// order-exempt: <reason>`) on the line above
  its `def` / `fn` or on that line; the reason is required, and a marker without one
  exempts nothing. Constants, fixtures and types go before the functions.
- **Tests:** helpers come first, in the same order, under a `# ---- helpers ----`
  comment (Rust: `// ---- helpers ----`). The tests follow in groups, each starting with a
  `# ---- <label> ----` comment; every test is in a group, no helper comes after the
  first test, and the group labels are alphabetical (`other tests` last). The checker
  enforces the helper, group and label rules. Inside a group the tests are **not** sorted: they stay in
  the order they were written (a reviewer checks this, the checker cannot), and a test
  file whose tests must run in a fixed order keeps that order, with the reason in the PR.
- **Reordering is its own commit.** A pure reorder never shares a commit with a change
  of behaviour, so a reviewer can read it with `git diff -w --color-moved`.

Check it with `uv run --no-sync python tools/check_function_order.py` (CI runs it too). The checker also reports malformed Rust (unbalanced braces, an unterminated string), and its own behaviour is pinned by `tests/test_check_function_order.py`.

## Local checks before every PR merge

Run the full default checks locally against the **latest PR commit** before
merging. From the repository root:

```shell
uv sync --locked --extra test --extra dev
uv run --no-sync python -m pytest tests -q -p no:cacheprovider
uv run --no-sync ruff check src/ tests/cffbps
uv run --no-sync mypy
uv lock --check
uv run --no-sync python tools/check_function_order.py
```

Always run Python through `uv run --no-sync` so the project environment is the
one that runs (a bare `python` or `pytest` may belong to another environment).
The suite includes the Python/Rust parity tests, so the compiled extension must
be current: after any change under `rust/`, rebuild it with
`uv sync --reinstall-package cffdrs --extra test --extra dev` (a plain `uv sync`
does not notice changes to Rust sources). If an older build left a stale
`src/cffdrs/_rust.cp3*` extension file in the source tree, delete it: it can
shadow the current abi3 extension.

When a PR changes anything under `rust/`, also run the Rust checks. There is no
`Cargo.toml` at the repository root, so every command passes
`--manifest-path rust/Cargo.toml`:

```shell
cargo fmt    --manifest-path rust/Cargo.toml --all --check
cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --manifest-path rust/Cargo.toml --workspace --no-deps
cargo test   --manifest-path rust/Cargo.toml -p cffbps-rust
cargo +<MSRV> check --manifest-path rust/Cargo.toml --workspace --all-targets --locked
```

`<MSRV>` is the `rust-version` declared in `rust/Cargo.toml` (currently 1.78).
Document any skips, failures, or unavailable platform checks; a failed suite does
not satisfy the passing-test checkbox below.

Rerun the checks after changes to the PR branch. Record the commands, Python
version, Rust toolchain, operating system, latest commit SHA, and pass/skip counts
in the PR description. Do not treat a run against an earlier commit as validation
of the current PR head.

## Rules for scientific and numerical changes

The Python package is the reference implementation: write and test a science
change in Python first, then port it to Rust in the same PR. Agents supply Rust
design and review, so the evidence is executable:

- **Goldens.** The files under `tests/cffbps/data/golden/` are the behavior
  oracle. Do not regenerate them to accommodate a refactor. A deliberate
  numerical change regenerates them with `tools/gen_fbp_goldens.py`, and the PR
  reviews the resulting diff line by line: take intended changes only, and
  discard last-digit platform noise.
- **Bit-level oracle.** If a change touches the scalar chain (`run()` and the
  stage modules in `rust/crates/cffbps-rust/src/`), record the oracle baseline
  **before** the change and run the comparison after it
  (`cargo test --manifest-path rust/Cargo.toml -p cffbps-rust --test characterization -- --ignored`;
  see the test's module docs). It must pass unchanged unless the change is a
  deliberate numerical one; in that case diff the old and new baselines, confirm
  that only the intended fields changed, and say so in the PR. The baseline lives
  in `rust/target/` and is never committed.
- **Backend differences.** A change to either backend runs
  `uv run --no-sync python tools/compare_backends.py` and compares the table with
  `docs/BACKEND_DIFFERENCES.md`. A new row or a changed count is either fixed or
  added to the register with a status and a reason.
- **Benchmarks.** A change to the scalar or grid hot paths runs the release
  benchmarks (`rust/crates/cffbps-rust/examples/bench_run.rs` and
  `tools/bench_rust_grid.py`) three times on a quiet machine and records all runs;
  the scalar pass must stay within +5% and the grid pass within +10% of the
  recorded baselines. Timings on a shared machine vary by about 10%: repeat, do not
  guess.
- **Experimental CuPy backend.** It is not held to parity and cannot be tested in
  CI. If a change touches it, say how it was checked (for example a NumPy stand-in)
  and whether a GPU run was done.
- Describe changes to scientific assumptions, units, outputs, missing-value
  behavior and regression risks in the PR. Add or update tests for cffdrs-owned
  behavior; avoid tests that only repeat behavior owned by a dependency.

## Large tests (manual, required for model changes)

Expensive tests are **never run automatically on GitHub** (the GitHub Actions
credits are limited and nothing is scheduled). Run them yourself, locally, against
the latest PR commit. They are required for a PR that changes `rust/**`,
`src/cffdrs/cffbps/**` or the differential-test tooling; documentation-only or
CI-only PRs mark them not applicable and say why.

- The Python-versus-Rust comparison (`tools/compare_backends.py`) described above.
  Record the number of scenario runs and whether any value difference appeared.
- When Rust logic under `rust/crates/cffbps-rust/src/` changes and `cargo-mutants`
  is available: mutation testing of the changed lines only, in a clean
  `git worktree` (it rewrites source files while it runs, so never run it in your
  working tree). Every missed mutant is killed by a new test or listed as an
  equivalent mutant with a one-line justification. Record the counts (caught,
  missed, unviable, timeouts).

No workflow runs these large tests (to spare Actions credits they are manual and
local, and nothing runs on a schedule), so they are run on a suitable machine and the
results are recorded in the PR.

## Dependency changes and lock files

- `pyproject.toml` is the only hand-edited Python dependency declaration.
  `uv.lock` is generated: never edit it. After any dependency change run
  `uv lock`, commit both files, and confirm `uv lock --check` passes. To update
  one package: `uv lock --upgrade-package <name>`; to refresh everything:
  `uv lock --upgrade`. Then run the full checks and review the lock diff (changed
  versions, added or removed packages, and sources).
- Rust dependencies are declared in the crate `Cargo.toml` files; `rust/Cargo.lock`
  is generated by Cargo (use `cargo update -p <name>`, never edit it by hand). It is
  committed and the MSRV check runs with `--locked`, so a lockfile change must be
  deliberate and reviewed, and the lockfile format must stay compatible with the MSRV.
  The `cffbps-rust` crate has no runtime dependencies and forbids `unsafe` code;
  adding a dependency to it needs an explicit reason in the PR. The minimum supported
  Rust version is declared once as `rust-version` in `rust/Cargo.toml`; raise it only
  after confirming `cargo +<new version> check --locked` passes, and say so in the PR.
- A numerical change after an upgrade is scientific behavior, not a tooling
  update: review it as such before merging.
- The supported Python versions are 3.11 to 3.14 (`requires-python >= 3.11`) and
  the PyO3 / rust-numpy version is pinned in `rust/crates/cffdrs-py/Cargo.toml`.
  Raising or lowering either is a minor-version change that needs the release
  workflow run described below.

## Pull requests and merge review

- Every branch merge requires a PR. Follow the target branch in the versioned
  routing above. Do not merge or fast-forward work branches directly into
  `master` or `pre_release`.
- Every PR description must include these checkboxes and the test record (the PR
  template at [`.github/pull_request_template.md`](../.github/pull_request_template.md)
  fills them in):

  ```markdown
  - [ ] Full default checks (and the Rust checks, when `rust/` changed) passed locally against this PR's latest commit.
  - [ ] Large tests were run locally against this PR's latest commit, or are not applicable (reason below).

  Commit SHA:
  Python / Rust toolchain / OS:
  Commands:
  Result (passed / skipped):
  Golden / oracle / differences-register changes:
  Large tests (cases, differences found) or not-applicable reason:
  Additional opt-in tests or limitations:
  ```

- Check a box only after the check passes on the latest commit. If a check was not
  run or did not pass, leave the box unchecked and explain why. **Do not merge a PR
  while a box is unchecked.** Reviewers must verify the recorded SHA and results
  before merging.
- Verify every automated review comment (Copilot, other agents) against the code
  before changing anything: reproduce the claim with a failing test first, fix
  valid findings in a small commit, and answer on the thread only with the owner's
  permission. A review comment may be wrong, or already covered by the
  differences register.
- A change that alters results (a numerical or missing-value rule, a golden, an
  oracle baseline) states exactly which outputs change and why, in the PR
  description and in the commit message.

The checkboxes are a reviewer-enforced rule, not a GitHub-required status check. A
checked box records the author's local result; it does not replace review of the
output or prove that other operating systems passed.

## GitHub Actions

`ci.yml` runs on every pull request and on pushes to `feature/**` and `master`: the
Rust checks (format, clippy, docs, tests, MSRV) and the Python job (pytest, ruff,
mypy). `release.yml` runs only on manual dispatch or on a pushed `v*` tag: it builds
the sdist and the platform wheels, smoke-tests them on every supported Python, and
runs `twine check`. Publishing to PyPI and TestPyPI is disabled in the workflow
(`if: false`) until the owner enables it, so a dispatch or tag cannot upload
anything. Nothing is scheduled.

Before a release, run `release.yml` by hand on the release branch and confirm that
every build and smoke-test job passes. Enabling the publish jobs, creating a tag and
publishing all require the owner's direct permission, one action at a time.

Local checks and the completed PR record remain required whether or not GitHub
Actions runs. Coding agents must obtain direct permission before dispatching the
release workflow or pushing release tags.
