#!/usr/bin/env bash
#
# Container entrypoint.
#
# Its one real job is file ownership: the container writes checkpoints, reports
# and splits.json into directories mounted from the host. If it ran as root,
# every one of those files would land on the host owned by root. So we align the
# in-container user with PUID/PGID (defaults 1000:1000) and drop privileges.
#
set -euo pipefail

PUID="${PUID:-1000}"
PGID="${PGID:-1000}"
APP_USER=trainstudio

log() { printf '[entrypoint] %s\n' "$*"; }

# ── Directories that must exist and be writable ──────────────────────────────
for d in "${TRAINSTUDIO_RUNS_DIR:-}" \
         "${XDG_CACHE_HOME:-/cache}" \
         "${HF_HOME:-/cache/huggingface}" \
         "${TORCH_HOME:-/cache/torch}" \
         "${MPLCONFIGDIR:-/cache/matplotlib}" \
         "${YOLO_CONFIG_DIR:-/cache/ultralytics}" \
         "${TRAINSTUDIO_HOME:-/opt/trainstudio-state}"; do
    [ -n "$d" ] && mkdir -p "$d" 2>/dev/null || true
done

# ── Network sanity ───────────────────────────────────────────────────────────
# BIND_ADDRESS is passed in for this check alone: it decides the host side of
# the port publish, which the container cannot otherwise see.
#
# On current Streamlit the browser opens its WebSocket relative to whatever
# origin it loaded the page from, so a remote browser works even when
# PUBLIC_HOSTNAME is wrong — only the URL Streamlit prints, and features that
# need an absolute link, go astray. Hence a note rather than a warning.
case "${BIND_ADDRESS:-127.0.0.1}" in
    127.0.0.1|localhost|::1|"") ;;
    *)
        log "note: port published on ${BIND_ADDRESS} with NO authentication in front."
        log "      Anyone who can reach it browses this host's filesystem and starts runs."
        log "      For shared access use the authenticated proxy instead:"
        log "          docker compose --profile proxy up -d"
        if [ "${STREAMLIT_BROWSER_SERVER_ADDRESS:-localhost}" = "localhost" ]; then
            log "note: PUBLIC_HOSTNAME is still 'localhost', so the URL logged below and"
            log "      any absolute link the app builds will point at the wrong machine."
        fi
        ;;
esac

if [ "$(id -u)" = "0" ]; then
    # Align the app user with the host user that owns the mounted volumes
    current_uid="$(id -u "${APP_USER}")"
    current_gid="$(id -g "${APP_USER}")"

    if [ "${PGID}" != "${current_gid}" ]; then
        log "setting the ${APP_USER} group to GID ${PGID}"
        groupmod -o -g "${PGID}" "${APP_USER}"
    fi
    if [ "${PUID}" != "${current_uid}" ]; then
        log "setting the ${APP_USER} user to UID ${PUID}"
        usermod -o -u "${PUID}" "${APP_USER}"
    fi

    # Only chown what we own the lifecycle of. The dataset mount is left alone:
    # it can be huge, and it is read-only as far as the application is concerned.
    # Note: the mounted host /home is deliberately NOT chowned — it is the
    # user's real home and already has the right ownership.
    for d in "${XDG_CACHE_HOME:-/cache}" \
             "${TRAINSTUDIO_HOME:-/opt/trainstudio-state}" \
             "${TRAINSTUDIO_RUNS_DIR:-}"; do
        [ -z "$d" ] && continue
        chown -R "${PUID}:${PGID}" "$d" 2>/dev/null || \
            log "warning: could not chown $d (continuing)"
    done

    # Fail loudly on the misconfiguration that is otherwise a bare EACCES deep
    # inside the app: $HOME set to a path this user cannot read.
    if ! gosu "${PUID}:${PGID}" test -r "${HOME:-/nonexistent}" 2>/dev/null; then
        log "WARNING: HOME=${HOME:-unset} is not readable by ${PUID}:${PGID}."
        log "         Set HOST_HOME in .env to your own home directory"
        log "         (e.g. HOST_HOME=/home/\$USER) and recreate the container."
    fi
    if [ -n "${TRAINSTUDIO_RUNS_DIR:-}" ] && \
       ! gosu "${PUID}:${PGID}" test -w "${TRAINSTUDIO_RUNS_DIR}" 2>/dev/null; then
        log "WARNING: TRAINSTUDIO_RUNS_DIR=${TRAINSTUDIO_RUNS_DIR} is not writable"
        log "         by ${PUID}:${PGID}. Set RUNS_DIR in .env to a mounted path."
    fi

    log "starting as ${APP_USER} (${PUID}:${PGID}): $*"
    exec gosu "${PUID}:${PGID}" "$@"
fi

# Already running as a non-root user (e.g. compose `user:` was set explicitly)
log "starting as uid $(id -u): $*"
exec "$@"
