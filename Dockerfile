# Container image for the interactive demo GUI (`pva serve`).
#
# Build:  docker build -t pva-demo .
# Run:    docker run --rm -p 8080:8080 pva-demo
#
# The demo has no authentication and is rate-limited rather than access
# controlled. Terminate TLS and apply access rules at a reverse proxy in front
# of this container.

FROM python:3.12-slim AS build

WORKDIR /build

# Copy only what the wheel build needs, so dependency layers stay cached.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir hatchling \
    && hatchling build -t wheel


FROM python:3.12-slim

# Run as an unprivileged user: the demo never needs to write to the filesystem
# outside of a per-request temporary directory.
RUN useradd --create-home --uid 10001 pva

COPY --from=build /build/dist/*.whl /tmp/

# Install the freshly built wheel together with its `web` extra, so the demo
# server ships from this source tree rather than from PyPI.
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[web]" \
    && rm -f /tmp/*.whl

USER pva
# The structured security logger appends to ./pva.log, so the working directory
# has to be writable by the unprivileged user.
WORKDIR /home/pva

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4).status == 200 else 1)"

CMD ["pva", "serve", "--host", "0.0.0.0", "--port", "8080"]
