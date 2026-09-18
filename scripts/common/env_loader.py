"""
scripts/common/env_loader.py

Shared credential-bootstrap utility used by every script in this project.

Usage:
    from scripts.common.env_loader import load_env_and_validate

    env = load_env_and_validate([
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ])
"""

import sys
from typing import Optional

from dotenv import find_dotenv, load_dotenv


def load_env_and_validate(required_vars: list[str]) -> dict[str, str]:
    """Load `.env` from the project root and validate required environment variables.

    Behaviour:
    - Calls ``dotenv.load_dotenv(find_dotenv(), override=False)`` so that variables
      already present in the shell environment are *not* overridden.
    - If ``find_dotenv()`` returns an empty string (no ``.env`` file found anywhere
      in the directory hierarchy), prints an error and calls ``sys.exit(1)`` before
      making any AWS API calls.
    - Checks each entry in *required_vars*; if any variable is absent from the
      environment or is set to an empty string, prints
      ``Missing required environment variable(s): <names>`` and calls
      ``sys.exit(1)``.
    - Never prints credential *values* — only variable names.

    Args:
        required_vars: Ordered list of environment variable names that must be
            present and non-empty after loading ``.env``.

    Returns:
        A ``dict`` mapping each name in *required_vars* to its resolved string value.

    Raises:
        SystemExit: With exit code ``1`` if ``.env`` is absent or any required
            variable is missing or empty.
    """
    import os  # imported here to keep the module-level namespace tidy

    # Locate and load the .env file.
    dotenv_path: Optional[str] = find_dotenv()
    if not dotenv_path:
        print(
            "Error: No .env file found. "
            "Create a .env file in the project root with the required credentials."
        )
        sys.exit(1)

    # override=False means shell-set variables take precedence over .env values.
    load_dotenv(dotenv_path=dotenv_path, override=False)

    # Validate that every required variable is present and non-empty.
    missing: list[str] = [
        name
        for name in required_vars
        if not os.environ.get(name, "").strip()
    ]

    if missing:
        # Print only names — never values — to avoid leaking credentials.
        print(f"Missing required environment variable(s): {', '.join(missing)}")
        sys.exit(1)

    return {name: os.environ[name] for name in required_vars}
