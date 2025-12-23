# Agent Guidelines and Known Issues

This document contains guidance for AI agents working on this project, including known issues and workarounds.

---

## OneDrive Sync Issues

### Problem Description

When this project is stored in a OneDrive-synced folder (`~/Library/CloudStorage/OneDrive-*/`), file operations can become extremely slow or hang. This is caused by:

1. **OneDrive sync daemon** intercepting all file system calls
2. **Extended attributes** being updated for sync status
3. **Index lock contention** when OneDrive and git both try to update metadata
4. **Network latency** if OneDrive is syncing to cloud during operations

### Symptoms

- Simple commands like `git status` or `ls` take 30+ seconds or timeout
- Commands go to background mode unexpectedly
- Git operations hang with no output
- File reads/writes are slow

### Solutions

#### Option 1: Pause OneDrive Syncing (Recommended)

1. **Via Menu Bar** (User action required):
   - Click the OneDrive icon in the macOS menu bar
   - Click the gear icon (Settings)
   - Select "Pause Syncing" → Choose duration (2 hours, 8 hours, 24 hours)

2. **Via System Settings**:
   - Open System Settings → Internet Accounts → OneDrive
   - Toggle off sync temporarily

**Note on Terminal Commands**: Microsoft does not provide a command-line interface for OneDrive on macOS. The following commands will QUIT OneDrive (not pause):

```bash
# Quit OneDrive (may auto-restart)
osascript -e 'quit app "OneDrive"'

# Force kill (will auto-restart via launchd)
killall OneDrive

# Prevent auto-restart (more aggressive)
launchctl unload ~/Library/LaunchAgents/com.microsoft.OneDrive*.plist 2>/dev/null
killall OneDrive
```

To restart after manual stop:
```bash
open -a OneDrive
```

The menu bar pause is the only reliable way to temporarily stop syncing without quitting.

#### Option 2: Work in Local Directory

For git-intensive operations, copy the repository to a local non-synced directory:

```bash
# Create a local working copy
cp -R "/Users/jesseandrews/Library/CloudStorage/OneDrive-TexasTechUniversity/Projects/Freeze and Flight" ~/Projects/freeze-flight-local

# Work in local copy
cd ~/Projects/freeze-flight-local

# After work is done, sync back
rsync -av --exclude='.git' ~/Projects/freeze-flight-local/ "/Users/jesseandrews/Library/CloudStorage/OneDrive-TexasTechUniversity/Projects/Freeze and Flight/"
```

#### Option 3: Use Git Worktree

Create a worktree in a non-synced location:

```bash
cd "/Users/jesseandrews/Library/CloudStorage/OneDrive-TexasTechUniversity/Projects/Freeze and Flight"
git worktree add ~/Projects/freeze-flight-worktree main
```

### When to Apply These Solutions

Apply these workarounds when:
- Git commands (status, add, commit, push) are timing out
- File operations take longer than 5 seconds
- Multiple background tasks accumulate without completing

### Detecting OneDrive Issues

Check if OneDrive is causing problems:

```bash
# Check if we're in an OneDrive directory
pwd | grep -q "CloudStorage/OneDrive" && echo "WARNING: OneDrive sync may slow operations"

# Check for OneDrive processes
pgrep -l OneDrive

# Check for git locks
ls -la .git/*.lock 2>/dev/null
```

### Agent Protocol for OneDrive Projects

When working on projects in OneDrive-synced directories:

1. **Set shorter timeouts** for file operations (5-10 seconds vs default 120 seconds)
2. **Batch file writes** - write multiple files in sequence rather than parallel
3. **Check for hangs** - if 3+ commands go to background, OneDrive is likely the cause
4. **Advise user** to pause OneDrive sync for git-intensive work
5. **Use simple commands** - avoid complex git operations when possible

---

## Git Operations Best Practices

### Commits

- Always use `--no-gpg-sign` if signing is causing issues
- Use short timeouts and be prepared for operations to hang
- If commit hangs, check for `index.lock` file

### Removing Lock Files

If git is locked due to interrupted operation:

```bash
rm -f .git/index.lock
rm -f .git/refs/heads/*.lock
```

**Warning**: Only remove lock files if no git operation is in progress.

---

## Project-Specific Notes

### Data File Locations

Large data files are in `data_work/` and `GIS_Data/`. These sync slowly and should not be modified during active OneDrive sync.

### Virtual Environment

The `.venv/` directory should be in `.gitignore` but may still sync via OneDrive. Consider moving it outside the project if experiencing issues.

---

---

## Git Repository Notes

**Important**: Check if the project is a git repository before running git commands:

```bash
# Check if .git exists
ls -la .git 2>/dev/null || echo "Not a git repository"

# Or check git status
git status 2>&1 | head -1
```

If the project is NOT a git repo and you need to initialize:

```bash
# Initialize new repo
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit"

# Add remote (if needed)
git remote add origin https://github.com/username/repo.git

# Push
git push -u origin main
```

---

---

## Bash Execution Issues

### Problem: All Commands Go to Background

In some sessions, all bash commands (even simple ones like `echo`) are sent to background execution and timeout. This appears to be a sandbox/environment issue.

**Symptoms**:
- Every bash command returns "Command running in background with ID: xxx"
- TaskOutput times out waiting for completion
- Even simple commands like `echo "test"` fail

**Workaround**:
1. File operations (Read, Write, Edit, Glob, Grep) still work
2. Provide user with manual commands to run in their terminal
3. Create shell scripts via Write tool that user can execute

**Manual Commands Template**:
```bash
# Git operations
cd "/path/to/project"
git init
git add .
git commit -m "message"
git push
```

---

## Revision History

| Date | Change |
|------|--------|
| 2025-12-23 | Initial documentation of OneDrive sync issues |
| 2025-12-23 | Added git repository notes |
| 2025-12-23 | Added bash execution troubleshooting |
