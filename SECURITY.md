# Security policy

## Supported versions

Only the newest release of Volura gets fixes. Volura checks for updates itself (Help > Check for
Updates); please update before you report a problem.

## Reporting a vulnerability

Please do **not** report security problems in public issues or Discussions.

Report them privately through
[private vulnerability reporting](https://github.com/Lexvdpoel/Volura/security/advisories/new)
(the **Security** tab of the repository, then **Report a vulnerability**). Describe what the
problem is, which version and operating system you use, and the steps or a file that show it.

You get an answer as soon as possible. When the problem is confirmed, it is fixed in a new release,
and you are credited in the advisory unless you prefer not to be.

## Scope

In scope: Volura itself, its installer and the files it reads and writes (`.volura`, STL, 3MF and
the other import formats), and the update check.

Third-party software that Volura uses has its own security process: report problems in Qt,
CuraEngine, manifold3d and the other components listed in `THIRD_PARTY_LICENSES.txt` to their own
projects.
