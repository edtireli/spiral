# Explicit prerequisites

Dependency and model acquisition now comes only from validated typed analyst
declarations or existing project dependency manifests. Natural-language domain
words, site names and model phrases no longer select Python packages, binaries
or model downloads. This change is specific to capability acquisition: other
product-profile and delivery classification rules still exist.

New analyst output uses these disjoint types:

- `python-package:REQUIREMENT`: a public-registry distribution requirement.
- `python-runtime:SPECIFIER`: a version constraint on the already selected engine
  interpreter, for example `>=3.11`. This is never a pip requirement or authority
  to install/select another interpreter.
- `node:PACKAGE`, `brew:CORE_FORMULA`, `ollama:MODEL`, `binary:NAME`: the existing
  bounded typed prerequisites, with unchanged broker permissions. A binary-only
  declaration never acquires a guessed formula.

The shared parser validates the complete list before callers can declare or
acquire anything. Each entry is capped at 160 UTF-8 bytes; there are at most 24
per deliverable and 64 combined. Invalid values are rejected, not truncated or
silently dropped. Python URLs, multiline pip options, formula taps/casks and
generated shell text are not declaration types. Values never supply certificate
code. The analyst's existing two-attempt correction path receives a validation
diagnostic before artifacts.json or dependency-manifest mutation; persistent
invalid output is a manifest-protocol failure, not a failed package install.

## Compatibility and existing projects

Unambiguous legacy `python:PACKAGE` entries remain readable. A legacy suffix that
also parses as a PEP 440 version literal is ambiguous and must be re-declared with
an explicit runtime/package type. No particular version or package name is
blacklisted: an explicitly typed numeric distribution remains a distribution
request. Syntactic validity does not prove that the registry contains a package.
Legacy untyped prose prerequisites are rejected rather than interpreted as new
install instructions. Existing metadata is revalidated when consumed for setup;
it is not silently converted into new authority.

Goal-only preflight still inspects and synchronizes existing manifests. Existing
bad requirements and historical capabilities receipts are not deleted or repaired
automatically. Existing Python pins remain unchanged. A bare requested name can
reuse an existing pin; a different explicit constraint, extra or marker requires
an explicit manifest reconciliation instead of silently rewriting it. This is a
conservative equality check, not a general dependency-constraint solver. These
conflicts are checked before earlier new declarations can mutate any manifest.

## Evidence and remaining limits

Runtime evidence records the exact selected engine executable, observed Python
version and specifier result. It does not certify another project's virtualenv.
Typed Python package presence checks an installed distribution's metadata version
against the requested specifier. That is not an import/behavior certificate.
Extras or environment markers are not independently certified by this lookup and
are never marked present on name alone; their existing manifest resolver path is
unchanged. Node's pre-existing presence probe checks a package directory, not full
semver or dependency-graph correctness.

Existing broker registry, lifecycle-script, filesystem and full-access restrictions
are unchanged. This patch does not repair the existing installer's mutable-project
interpreter trust and rollback limitations, nor provide a new isolated resolver.
No real model inference or online package installation is used by the new tests.

## Verification

The focused prerequisite/capability/planner suite passed 89 tests, independently
repeated by a second reviewer. The final complete engine suite passed **1,103 tests
with one intentional offline live-inference skip in 87.23 seconds**, using the
existing engine Python 3.11.4 venv. Log:
`/Users/edt/spiralChat/build/autonomy-engine-explicit-prerequisites-tests-final.log`.
The earlier full invocation used the bundled-Python host test venv, which lacked
the `build` module, and found one old pipeline fixture expecting prose-driven
dependency declaration. That fixture now supplies an explicit package declaration;
the earlier log is retained. No production test assertion was disabled. The final
run also emitted nonfatal Playwright teardown/pending-task diagnostics; it exited
zero. These are offline suite results, not proof of autonomous completion.
