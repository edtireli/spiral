# Managed shell and Python environment boundaries

SpiralChat's full-access approval widens the filesystem ceiling. It does not grant
arbitrary shell networking. Managed shell commands still require OS isolation;
network egress, ambient credentials, Git mutations and protected runtime paths
remain constrained by their existing policies. Standalone full-access behavior
is unchanged. The worker prompt, CLI notice and direct broker feedback now describe
that distinction. `BrokerResult.network` is authoritative process metadata, not
text inferred from command output or reread from a workspace-mutable audit file.
Pre-launch refusals report `not-executed`.
The optional standalone unsandboxed workspace fallback reports `not-isolated`,
not `denied`; absence of enforcement does not grant network authority. Managed
runs never enter this fallback.

Typed web, anonymous browser research, download and registry-install requests remain
the supported network paths within their existing permissions. Approved wheel
downloads can be followed by a local, network-disabled installation. Such a route
must still fulfill the user's actual environment and verification requirements;
success in a different environment is not equivalent.

## Existing installer locations and unresolved limits

`ASK: install python PACKAGE` has no destination argument. It installs into
`PROJECT/.spiral/tooling/python` and adds that environment's `bin` to subsequent
brokered command PATHs. Automatic declared-dependency synchronization uses the
different `PROJECT/.spiral/dependency-cache/python/venv` environment. Neither
automatically satisfies a request to produce another exact environment path.
The typed install result now returns its actual target and interpreter explicitly.

Custom Python environment targeting is deliberately not implemented by this change.
The existing installer launches the managed environment's mutable `bin/python` for
networked pip operations. Accepting an arbitrary preexisting project environment
would widen that unresolved trust problem. Existing failure cleanup can also invoke
pip uninstall in a preexisting environment. These preexisting execution/rollback
risks are not fixed or expanded by the truthful-feedback change.

A future target-capable broker must use a trusted, isolated installer rather than
executing project-controlled Python/pip with network authority, bind the destination
and site-packages paths against symlink/identity escape, preserve prior packages on
failure, and record exact target installation and subsequent execution evidence.
No new shell networking, arbitrary target interpreter or approval bypass is added.
