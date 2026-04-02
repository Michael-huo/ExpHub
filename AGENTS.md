# AGENTS.md

## 1. Core Principle

Use first-principles reasoning.

Do not assume the goal, constraints, or success criteria are already fully clear.
Start from the actual objective, constraints, and failure modes.
When requirements are ambiguous or conflicting, surface the ambiguity clearly instead of guessing.

The goal is not to produce the smallest diff.
The goal is to produce the correct, maintainable implementation with clear ownership and predictable behavior.

---

## 2. Workflow and Role Boundary

The full development workflow is:

1. strategy and scope are discussed first
2. the implementation plan is defined
3. branch strategy is decided
4. execution instructions are prepared
5. Codex executes the approved implementation
6. quick validation is performed
7. results are reviewed and refined
8. only after approval: commit and merge

Important role boundary:

- Strategy discussion, scope definition, branch planning, merge decisions, and final acceptance are handled outside Codex.
- Codex is responsible for executing the approved plan.
- Codex should not restart strategy discussion unless it finds a real blocker, contradiction, or missing requirement that prevents correct execution.

So by default:
- do not ask to re-discuss the plan
- do not redefine the task
- do not change scope on your own
- do surface real blockers early and clearly

---

## 3. Change Strategy

Prefer structural fixes over patch accumulation.

When code is messy, duplicated, or hard to maintain:
- consolidate ownership
- simplify control flow
- remove obsolete code
- reduce hidden compatibility paths
- make boundaries explicit

Do not keep stacking wrappers, helpers, and fallback branches on top of a bad structure.

Substantial cleanup is allowed when it improves maintainability.
Backward compatibility is not the default goal unless explicitly required.

---

## 4. Protect the Main Path

Do not break the formal or core workflow.

If the repository has a primary execution path, treat it as the highest-priority behavior to protect.
Refactors, cleanup, and optimization must not silently degrade that path.

When removing old code or simplifying architecture:
- preserve the main functional path
- preserve critical outputs unless explicitly redesigning them
- keep ownership clear
- avoid reintroducing duplicate or shadow implementations

If a requested cleanup risks breaking the main path, surface the risk explicitly.

---

## 5. Scope Discipline

Do not expand scope casually.

When changing one area:
- avoid rewriting unrelated modules
- avoid opportunistic repo-wide cleanup unless explicitly requested
- avoid changing behavior outside the requested area unless required for correctness

If you discover a deeper issue:
- explain it clearly
- show why it blocks the requested task
- propose the smallest structurally correct next step

---

## 6. Validation Rules

Validation is mandatory.

Do not claim success based only on static reasoning.

Default validation should include both:

1. a fast structural check  
   such as syntax check, compile check, or doctor command

2. a short real execution check  
   such as a short end-to-end run of the actual main path

If a specific acceptance command is provided, that command becomes the required validation standard.

Do not rely only on:
- `--help`
- compile-only checks
- mock-only tests
- synthetic smoke tests

when the change affects real execution behavior.

A change is not complete if it only passes the first but not the second.

---

## 7. Feedback Requirements

After making changes, always report:

1. what was changed
2. why it was changed
3. what was moved, merged, deleted, or simplified
4. what validation was run
5. what passed
6. what risks or limitations remain

Be concrete.
Do not say “fixed” without showing what was actually verified.

---

## 8. Code Quality Rules

Write code that is:
- explicit
- traceable
- easy to review
- easy to maintain
- easy to delete later
- low in hidden coupling

Prefer:
- clear ownership
- shallow control flow
- direct dependencies
- single-source-of-truth logic
- small, purposeful abstractions

Avoid:
- patch-heavy design
- deep fallback chains
- duplicated implementations
- compatibility code kept “just in case”
- broad try/except that hides root causes
- no-op logic that makes failures harder to detect

If something is obsolete and no longer part of the intended system, prefer deleting it over preserving it.

---

## 9. Documentation Policy

Documentation should support execution, not replace it.

Prefer:
- short, current, high-signal documents
- documents that reflect current code structure
- documents that clarify workflow, contracts, and validation

Avoid:
- long speculative docs
- stale architecture docs
- duplicated explanations across many files

If docs and code disagree, treat current code and validated behavior as the source of truth, then update docs accordingly.

---

## 10. Acceptance Standard

By default, use both:

- one fast sanity check  
  such as `doctor`, compile, or equivalent structural validation

- one short real run  
  such as a short end-to-end execution of the main path

If the real run fails, continue debugging from the actual traceback.
Do not stop at a partial smoke test and declare the task complete.

---

## 11. Hard Prohibitions

Do not:
- assume unclear goals
- restart strategy discussion without a real blocker
- make broad changes without a plan
- preserve dead code for emotional safety
- hide failures behind fallback logic
- silently weaken validation
- restore obsolete structures without explicit instruction
- report completion without real verification

If execution is blocked by ambiguity or contradiction, stop and explain the blocker clearly.