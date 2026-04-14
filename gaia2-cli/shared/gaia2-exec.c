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
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *ALLOWED_TOOLS[] = {
    "calendar", "contacts", "emails", "messages", "chats",
    "rent-a-flat", "city", "cabs", "shopping", "cloud-drive",
    NULL
};

int main(int argc, char *argv[]) {
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

    /* Build path to real tool */
    char path[256];
    snprintf(path, sizeof(path), "/usr/local/bin/%s", name);

    /* Drop to gaia2 user (effective UID from setuid bit) */
    if (setgid(getegid()) != 0 || setuid(geteuid()) != 0) {
        perror("gaia2-exec: setuid/setgid");
        return 1;
    }

    /* Set state directory */
    setenv("GAIA2_STATE_DIR", "/var/gaia2/state", 1);

    /* Re-inject libfaketime only when faketime is configured (LD_PRELOAD is
     * stripped by kernel for setuid binaries, so we must re-set it).
     * Unset FAKETIME so the file takes precedence (allows dynamic time changes). */
    if (getenv("FAKETIME_TIMESTAMP_FILE") || access("/tmp/faketime.rc", F_OK) == 0) {
        setenv("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1", 1);
        unsetenv("FAKETIME");
        setenv("FAKETIME_TIMESTAMP_FILE", "/tmp/faketime.rc", 1);
        setenv("FAKETIME_NO_CACHE", "1", 1);
    }

    /* Exec the real tool */
    argv[0] = path;
    execv(path, argv);
    perror("gaia2-exec: execv");
    return 1;
}
