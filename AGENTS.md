# Repository Working Rules

Before modifying, creating, renaming, or deleting any project files:

1. Run `git status`.
2. Determine the current branch using `git branch --show-current`.
3. Never discard or overwrite existing uncommitted changes.
4. Run `git fetch origin`.
5. If the working tree is clean, synchronize the current branch using:
   `git pull --rebase origin <current-branch>`
6. If there are uncommitted changes that make synchronization unsafe, stop and tell me before proceeding.
7. If a merge or rebase conflict occurs, stop and explain the conflict instead of blindly resolving it.
8. Only begin implementing my requested code changes after synchronization succeeds.

Never use destructive Git commands such as:
- `git reset --hard`
- `git clean -f`
- `git checkout -- .`
- `git restore .`

unless I explicitly ask you to.

After completing a coding task:

1. Run `git status`.
2. Tell me which files were changed.
3. Do not commit or push unless I explicitly ask you to.