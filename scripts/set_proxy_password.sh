#!/usr/bin/env bash
#
# Set the username and password for the reverse proxy that fronts TrainStudio.
#
#   ./scripts/set_proxy_password.sh                 # prompts for both
#   ./scripts/set_proxy_password.sh alice 'secret'  # non-interactive
#
# Exists because the escaping is a trap. A bcrypt hash is full of `$`, and
# Compose interpolates `$`-sequences when it reads .env: a hash pasted verbatim
# reaches Caddy with `$2y$12$eNxx` turned into `$2y$12` — still a plausible
# looking string, so nothing errors. The proxy starts, serves the login box and
# rejects every password, with no clue in any log. So each `$` is written
# doubled here, which Compose collapses back to one.
#
set -euo pipefail

cd "$(dirname "$0")/.."
ENV_FILE=.env

[ -f "$ENV_FILE" ] || { echo "No .env yet — copy it first: cp .env.example .env" >&2; exit 1; }

USER_NAME="${1:-}"
PASSWORD="${2:-}"

if [ -z "$USER_NAME" ]; then
    read -rp "Username: " USER_NAME
fi
if [ -z "$PASSWORD" ]; then
    read -rsp "Password: " PASSWORD; echo
    read -rsp "Repeat:   " PASSWORD2; echo
    [ "$PASSWORD" = "$PASSWORD2" ] || { echo "The passwords do not match." >&2; exit 1; }
fi
[ -n "$USER_NAME" ] && [ -n "$PASSWORD" ] || { echo "Both are required." >&2; exit 1; }

# bcrypt, because that is what Caddy's basic_auth expects. htpasswd ships with
# apache2-utils; the caddy image itself is the fallback so this works on a host
# with nothing installed but Docker.
if command -v htpasswd >/dev/null 2>&1; then
    HASH="$(htpasswd -nbB -C 12 "$USER_NAME" "$PASSWORD" | cut -d: -f2)"
elif command -v docker >/dev/null 2>&1; then
    HASH="$(docker run --rm caddy:2-alpine caddy hash-password --plaintext "$PASSWORD")"
else
    echo "Need either htpasswd (apt install apache2-utils) or docker." >&2
    exit 1
fi

case "$HASH" in
    '$2'*) ;;
    *) echo "That does not look like a bcrypt hash: $HASH" >&2; exit 1 ;;
esac

ESCAPED="$(printf '%s' "$HASH" | sed 's/\$/$$/g')"

python3 - "$ENV_FILE" "$USER_NAME" "$ESCAPED" <<'PY'
import re, sys
path, user, escaped = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path).read()
for key, value in (("BASIC_AUTH_USER", user), ("BASIC_AUTH_HASH", escaped)):
    line = f"{key}={value}"
    pattern = re.compile(rf"^{key}=.*$", re.M)
    text, n = pattern.subn(lambda _m, l=line: l, text)
    if not n:
        text = text.rstrip("\n") + f"\n{line}\n"
open(path, "w").write(text)
PY

echo "Updated .env for user '$USER_NAME'."
echo "Apply it with:  docker compose --profile proxy up -d"
