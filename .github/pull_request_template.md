## What does this change?

<!-- A short description of the change and why it is needed. -->

Closes #<!-- issue number, or link the idea in Discussions -->

## How was it tested?

<!-- What you tried by hand, and which tests you added or ran. -->

## Screenshots

<!-- For a change you can see: before and after. Delete this section if it does not apply. -->

## Checklist

- [ ] I read [CONTRIBUTING.md](https://github.com/Lexvdpoel/Volura/blob/main/CONTRIBUTING.md)
- [ ] The whole test suite passes: `QT_QPA_PLATFORM=offscreen python -m pytest -q -n auto tests`
- [ ] New or changed behaviour has tests
- [ ] Every new text the user sees goes through `tr()` (with a Dutch translation in `volura/locale/nl.py`, or a note here that it is missing)
- [ ] A new ready-made part is one closed volume and is in `tests/test_parts_closed.py`
- [ ] `README.md` and `docs/website/manual.html` (English and Dutch) describe what users will notice
- [ ] The code, models and images in this pull request are mine to share under the MIT license
