#!/bin/sh
# psql against the throwaway cluster: ./pgq.sh -c "SELECT ..."
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
exec "$PGBIN/psql" -h "${ROOT:-/tmp/pgscan}" -p "${PORT:-54329}" -U postgres -d scan -X -At "$@"
