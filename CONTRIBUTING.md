# Contributing to Volura

Thank you for wanting to help! Volura is a 3D modelling program for 3D printing with the
push/pull workflow of SketchUp, and every bit of help makes it better: a bug report, an idea, a
translation, a fix or a new ready-made part.

Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first. Everyone who takes part is expected to
follow it.

## Questions, ideas and bugs

- **A question** ("how do I ...?"): ask it in the
  [Q&A of Discussions](https://github.com/Lexvdpoel/Volura/discussions/categories/q-a).
- **An idea** for a feature or a new part: post it in
  [Ideas](https://github.com/Lexvdpoel/Volura/discussions/categories/ideas). Upvote the ideas you
  want; the most wanted ones are picked up first.
- **A bug**: open a [bug report](https://github.com/Lexvdpoel/Volura/issues/new/choose). Give the
  version (Help > About Volura), your operating system and the steps that lead to the problem. A
  small `.volura` file and, after a crash, the `volura.log` next to the program help a lot.
- **A security problem**: do not post it in public. Contact the maintainer,
  [@Lexvdpoel](https://github.com/Lexvdpoel), privately through the contact details on that
  GitHub profile.

## Before you write code

For anything bigger than a small fix, open an idea or an issue first and say what you plan to do.
That way we can agree on the approach before you spend time on it, and two people do not build the
same thing.

## Setting up

You need Python 3.10 or newer (the builds use 3.12).

```sh
git clone https://github.com/Lexvdpoel/Volura.git
cd Volura
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
.venv/bin/python main.py                       # Windows: .venv\Scripts\python main.py
```

For **File > Slice** you also need CuraEngine and the printer profiles. They are not part of the
source tree; fetch them once with:

```sh
python tools/fetch_curaengine.py
python tools/fetch_orca_profiles.py
```

## Running the tests

```sh
QT_QPA_PLATFORM=offscreen python -m pytest -q -n auto tests    # Windows: $env:QT_QPA_PLATFORM = "offscreen"
```

The tests need no display and never touch your own settings. All of them must pass before a
change is merged; the GitHub Actions build runs them on Windows and Linux for every pull request.
Run one file while you work, for example `python -m pytest -q tests/test_pegboard.py`, and the
whole suite once at the end.

## How the code is written

- **Match the code around you**: the same naming, the same density of comments, the same idioms.
  Read a module next to the one you change before you start.
- **Geometry stays apart from the interface.** A module that makes a part (such as
  `volura/wall_hook.py`) has no Qt in it; it returns a `manifold3d.Manifold` and can be tested on
  its own. The dialog lives in `volura/part_dialogs.py`, the wiring in `volura/window_*.py` and
  `volura/app.py`.
- **Every part is one closed volume.** `tests/test_parts_closed.py` builds every ready-made part
  and checks that it comes back closed from the model; add your part to it.
- **Plain English** in names, docstrings, comments and messages. Docstrings say what something is
  for and why it is done this way, in sentences.
- **Every text the user sees** goes through `tr()`, and gets a Dutch translation in
  `volura/locale/nl.py`; `tests/test_i18n.py` checks that none is missing. Not Dutch-speaking?
  Leave the translation out and say so in the pull request; it will be added.
- **Platform independent**: no drive letters, no `os.startfile`; `tests/test_portability.py`
  checks it.
- **Tests come with the change**: a new feature gets tests that show it works, a bug fix a test
  that fails without the fix.

## Documentation

A change that users notice is described where they look for it:

- `README.md`: the feature list.
- `docs/website/manual.html`: the manual, in English and Dutch (`lang="en"` and `lang="nl"`).
- `docs/website/features.html`: the feature overview, when it is worth mentioning there.

The changelog is written when a release is made; you do not need to edit `CHANGELOG.md`.

## Commits and pull requests

1. Fork the repository and make a branch for your change.
2. Keep a pull request to one subject; several small ones are easier to review than one big one.
3. Write commit messages in English: a short first line that says what changes ("Pegboard: real
   L-hooks"), an empty line, and then why, if that is not obvious.
4. Open the pull request against `main`, describe what you changed and how you tested it, and link
   the issue or idea it belongs to.
5. Wait for the build to pass and for a review. Changes may be asked for; that is normal and
   meant to help.

## License

Volura is under the [MIT license](LICENSE). By contributing, you agree that your contribution is
published under that same license. Only contribute code, models and images that you have the
right to share.
