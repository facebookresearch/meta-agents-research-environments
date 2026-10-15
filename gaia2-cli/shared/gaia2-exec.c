/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * This source code is licensed under the terms described in the LICENSE file in
 * the root directory of this source tree.
 */
/*
 * gaia2-exec: setuid wrapper for GAIA2 CLI tools.
 *
 * This binary is installed as setuid-gaia2 so that the agent user can run
 * GAIA2 CLI tools without direct access to the state files.
 *
 * Usage: symlink this binary as the tool name (e.g. "calendar").
 * It resolves the real tool from /usr/local/bin/<basename> and execs it
 * as the gaia2 user with GAIA2_STATE_DIR set to /var/gaia2/state.
 *
 * Only whitelisted tool names are allowed to prevent arbitrary command execution.
 *
 * The caller is the (untrusted) agent user, so the inherited environment is
 * caller-controlled. Several CLIs pick a filesystem root (GAIA2_FS_ROOT), a
 * state dir (GAIA2_STATE_DIR) or a Python import path (PYTHON*) straight out of
 * the environment, and the dynamic loader honours LD_* — so the agent could set
 * GAIA2_FS_ROOT=/var/gaia2 and read the oracle scenario as gaia2, or set
 * PYTHONPATH to run arbitrary code as gaia2. We therefore wipe the environment
 * and rebuild it from a fixed allowlist rather than trusting what we inherit.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Directory the real tools live in. Overridable at COMPILE time only (never
 * from the environment) so a non-setuid unit test can point it at a stub. */
#ifndef GAIA2_EXEC_TOOL_DIR
#define GAIA2_EXEC_TOOL_DIR "/usr/local/bin"
#endif

static const char *ALLOWED_TOOLS[] = {
    "calendar", "contacts", "emails", "messages", "chats",
    "rent-a-flat", "city", "cabs", "shopping", "cloud-drive",
    NULL
};

/* Caller-supplied values we carry across the environment reset. These influence
 * only locale/timezone, never a filesystem root, state dir or code path. */
static const char *PRESERVE_VARS[] = {
    "TZ", "LANG",
    "LC_ALL", "LC_CTYPE", "LC_NUMERIC", "LC_TIME", "LC_COLLATE",
    "LC_MONETARY", "LC_MESSAGES", "LC_PAPER", "LC_NAME", "LC_ADDRESS",
    "LC_TELEPHONE", "LC_MEASUREMENT", "LC_IDENTIFICATION",
    NULL
};

#define MAX_PRESERVE 32

int main(int argc, char *argv[]) {
    (void)argc;
    /* Get basename of argv[0] (the symlink name) */
    const char *name = strrchr(argv[0], '/');
    name = name ? name + 1 : argv[0];

    /* Verify tool is in allowlist */
    int allowed = 0;
    for (const char **t = ALLOWED_TOOLS; *t; t++) {
        if (strcmp(name, *t) == 0) {
            allowed = 1;
            break;
        }
    }
    if (!allowed) {
        fprintf(stderr, "gaia2-exec: tool '%s' not allowed\n", name);
        return 1;
    }

    /* Build path to real tool before we touch the environment. */
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", GAIA2_EXEC_TOOL_DIR, name);

    /* Snapshot the allowlisted env values (and whether faketime was requested)
     * before clearing the environment. strdup so clearenv() cannot free them. */
    char *saved_vals[MAX_PRESERVE];
    const char *saved_keys[MAX_PRESERVE];
    int n_saved = 0;
    for (const char **k = PRESERVE_VARS; *k && n_saved < MAX_PRESERVE; k++) {
        const char *v = getenv(*k);
        if (v) {
            char *dup = strdup(v);
            if (!dup) {
                perror("gaia2-exec: strdup");
                return 1;
            }
            saved_keys[n_saved] = *k;
            saved_vals[n_saved] = dup;
            n_saved++;
        }
    }
    int faketime_env_requested = getenv("FAKETIME_TIMESTAMP_FILE") != NULL;

    /* Drop to gaia2 user (effective UID from setuid bit) */
    if (setgid(getegid()) != 0 || setuid(geteuid()) != 0) {
        perror("gaia2-exec: setuid/setgid");
        return 1;
    }

    /* Wipe the caller-controlled environment and rebuild a known-safe one.
     * This drops GAIA2_FS_ROOT, PYTHON*, LD_* and everything else the agent
     * might have set to redirect the tool at the oracle state or inject code. */
    if (clearenv() != 0) {
        fprintf(stderr, "gaia2-exec: clearenv failed\n");
        return 1;
    }

    /* Fixed PATH — never the caller's. */
    setenv("PATH", "/usr/local/bin:/usr/bin:/bin", 1);
    /* gaia2's home, not the agent's. */
    setenv("HOME", "/var/gaia2", 1);
    /* State directory the tools operate on. */
    setenv("GAIA2_STATE_DIR", "/var/gaia2/state", 1);

    /* Restore the locale/timezone allowlist. */
    for (int i = 0; i < n_saved; i++) {
        setenv(saved_keys[i], saved_vals[i], 1);
        free(saved_vals[i]);
    }

    /* Re-inject libfaketime only when faketime is configured (LD_PRELOAD is
     * stripped by the kernel for setuid binaries and we just cleared it, so we
     * must re-set it). Unset FAKETIME so the file takes precedence (allows
     * dynamic time changes). DONT_FAKE_MONOTONIC keeps short-lived tools'
     * timers on the real monotonic clock, matching the agent runtime. */
    if (faketime_env_requested || access("/tmp/faketime.rc", F_OK) == 0) {
        setenv("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1", 1);
        unsetenv("FAKETIME");
        setenv("FAKETIME_TIMESTAMP_FILE", "/tmp/faketime.rc", 1);
        setenv("FAKETIME_NO_CACHE", "1", 1);
        setenv("DONT_FAKE_MONOTONIC", "1", 1);
    }

    /* Exec the real tool */
    argv[0] = path;
    execv(path, argv);
    perror("gaia2-exec: execv");
    return 1;
}
