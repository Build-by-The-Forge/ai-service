import os

# Tests run as the "testing" environment: service secrets that are not set
# explicitly are generated per process (app.core.config). Must run before
# app.core.config is first imported.
os.environ.setdefault("APP_ENVIRONMENT", "testing")
