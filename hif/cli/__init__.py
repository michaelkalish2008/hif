"""The hif command surface.

LAYOUT RULE, and it is the whole rule: one module per command, named for the
command; shared infrastructure prefixed with `_`. `hif profile` lives in
profile.py, `hif batch` in batch.py, and anything two commands need is in an
underscore module. "Where does this go?" has one answer.

It did not before. `app` was built in cli_base.py while all seven commands sat
in a single 1748-line cli.py, so the module named *base* owned the app object
and the module named *cli* owned the commands — and cli_base also parsed .env,
which is neither. There was no rule to state, which is how it sprawled.

Commands register by importing this package: each command module decorates
with the `app` built in _app.py, so the imports below are the registration.
They look unused and are not — removing one removes a command.

`pyproject.toml` names `hif.cli:app` as the entry point, so `app` is
re-exported here and that is the only public name.
"""

from hif.cli._app import app

# Registration by import. Order sets the order commands appear in --help.
from hif.cli import profile as _profile      # noqa: F401
from hif.cli import models as _models        # noqa: F401
from hif.cli import doctor as _doctor        # noqa: F401
from hif.cli import compare as _compare      # noqa: F401
from hif.cli import render as _render        # noqa: F401
from hif.cli import schema as _schema        # noqa: F401
from hif.cli import batch as _batch          # noqa: F401
from hif.cli import config as _config_cmd    # noqa: F401

__all__ = ["app"]
