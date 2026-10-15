# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# gaia2-tls-env.sh — sourced (as root) by every gaia2-init-entrypoint.sh, which
# calls sanitize_tls_env before it starts any Gaia2 or agent process.
#
# Launchers that forward the host environment (e.g. apptainer) can carry CA
# bundle variables naming files that only exist on the host, such as a login
# node's SSL_CERT_FILE=/var/facebook/x509_identities/tls.crt. Every process in
# the container inherits them, and the common TLS clients refuse to start:
#   - httpx (SSL_CERT_FILE) raises FileNotFoundError, which kills gaia2-eventd
#     while litellm (1.105.0 builds an httpx client at import) loads the judge
#   - requests (REQUESTS_CA_BUNDLE) raises OSError on its first HTTPS request
#   - curl (CURL_CA_BUNDLE) exits with code 77
# sanitize_tls_env points such a variable at the image's CA bundle, or unsets
# it, before anything starts. A bundle the runner mounts (GAIA2_CA_BUNDLE, at
# /etc/ssl/certs/gaia2-host-ca-bundle.crt) that the gaia2 and agent users can
# read is kept as is. Node only warns about a bad NODE_EXTRA_CA_CERTS, but
# unsetting it lets the OpenClaw `oc` wrapper fall back to the TLS proxy's CA.

GAIA2_IMAGE_CA_BUNDLE="${GAIA2_IMAGE_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"


# True when $1 is a PEM certificate bundle that every process inheriting the
# variable can read. Root reads anything, so as root ask the gaia2 and agent
# users themselves (a host bundle mounted with mode 0660 is unreadable to
# them).
ca_bundle_usable() {
    local path="$1" user

    [ -f "$path" ] || return 1
    grep -qsE -- '-----BEGIN ([A-Z0-9]+ )*CERTIFICATE-----' "$path" || return 1
    [ "$(id -u)" = 0 ] || return 0
    for user in gaia2 agent; do
        id "$user" > /dev/null 2>&1 || continue
        su -s /usr/bin/bash "$user" -c "test -r $(printf '%q' "$path")" 2> /dev/null \
            || return 1
    done
}


sanitize_tls_env() {
    local var value

    # These replace the default trust store, so fall back to the image's.
    for var in SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do
        value="${!var:-}"
        if [ -z "$value" ] || ca_bundle_usable "$value"; then
            continue
        fi
        if ca_bundle_usable "$GAIA2_IMAGE_CA_BUNDLE"; then
            export "$var=$GAIA2_IMAGE_CA_BUNDLE"
            echo "[gaia2-init] $var=$value is not a CA bundle the gaia2 and agent" \
                "users can read; using $GAIA2_IMAGE_CA_BUNDLE (mount a" \
                "world-readable bundle with GAIA2_CA_BUNDLE)" >&2
        else
            unset "$var"
            echo "[gaia2-init] $var=$value is not a CA bundle the gaia2 and agent" \
                "users can read; unset it" >&2
        fi
    done

    # httpx uses this directory only when SSL_CERT_FILE is unset.
    value="${SSL_CERT_DIR:-}"
    if [ -n "$value" ] && [ ! -d "$value" ]; then
        unset SSL_CERT_DIR
        echo "[gaia2-init] SSL_CERT_DIR=$value is not a directory in the container;" \
            "unset it" >&2
    fi

    # Node adds these to its built-in store, which still applies without them.
    value="${NODE_EXTRA_CA_CERTS:-}"
    if [ -n "$value" ] && ! ca_bundle_usable "$value"; then
        unset NODE_EXTRA_CA_CERTS
        echo "[gaia2-init] NODE_EXTRA_CA_CERTS=$value is not a CA bundle the" \
            "gaia2 and agent users can read; unset it" >&2
    fi
}
