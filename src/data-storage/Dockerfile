FROM postgres:18-bookworm

# Single RUN layer: install deps, refresh certs, build pg_partman from source, clean up.
# ca-certificates + update-ca-certificates must run before git clone to avoid SSL failures.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    postgresql-18-pgvector \
    postgresql-18-pgtap \
    build-essential \
    git \
    postgresql-server-dev-18 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --branch v5.4.3 --depth 1 https://github.com/pgpartman/pg_partman.git /tmp/pg_partman \
    && cd /tmp/pg_partman \
    && make NO_BGW=1 \
    && make install NO_BGW=1 \
    && rm -rf /tmp/pg_partman

COPY config/postgresql.conf /etc/postgresql/postgresql.conf
COPY initdb/ /docker-entrypoint-initdb.d/
# scripts/ is mounted at runtime via docker-compose — not baked into the image
