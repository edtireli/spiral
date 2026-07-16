---
name: dependency-medic
description: Build toolchain and dependency failures — gradle, JDK, SDK versions, AGP, maven resolution, kotlin plugin, npm, pip. Use when build errors mention versions, resolution, class file versions, AAR metadata, plugins, or licenses.
---
# Dependency medic playbook

## Prime directives
1. PROBE before prescribing. Observe the machine; never trust remembered versions:
   `java -version` · `./gradlew --version` · `ls ~/Library/Android/sdk/platforms`
   · `ls ~/Library/Android/sdk/build-tools` · `cat gradle/wrapper/gradle-wrapper.properties`
2. Pin to versions OBSERVED on this machine (installed SDKs, cached artifacts).
   Never "upgrade to latest" from memory — that is where hallucination lives.
3. One change → re-run gate → read again. Dependency errors are sequential;
   fixing two at once hides which one worked.
4. Make the SMALLEST version move that satisfies the constraint, preferring
   moves within the already-downloaded set.

## Signature table (gradle/android)
| Error contains | Meaning | Fix |
|---|---|---|
| `Unsupported class file major version N` | JVM newer than gradle supports | run gradle on an older JDK: `org.gradle.java.home=<path>` in gradle.properties (Android Studio bundles one at `/Applications/Android Studio.app/Contents/jbr/Contents/Home`) |
| `requires JVM XX` / `invalid source release` | JVM older than required | point java home at a newer JDK (same key) |
| `AAR metadata ... compileSdk XX or higher` | a library outgrew compileSdk | raise compileSdk to XX **or** downgrade that library below the requirement |
| `Duplicate class X found in modules A and B` | two artifacts ship one class | `exclude(group=..., module=...)` on one dependency, or align both to one BOM |
| `Could not resolve <coordinate>` | typo'd coordinate, missing repo, or offline | verify exact coordinate spelling; ensure `google()` + `mavenCentral()` in settings repositories; check network |
| `plugin ... was not found` | plugin id/version wrong or repo missing | pin known id+version in root build file; `gradlePluginPortal()` present |
| `licence`/`license ... not accepted` | SDK licenses unaccepted | licenses live in `<sdk>/licenses/`; accepting requires sdkmanager — surface to human if absent |
| `Failed to find platform android-XX` | platform not installed, auto-download off | use an installed platform from the probe, or allow AGP auto-download |
| `Unresolved reference` in *.gradle.kts line 1-5 | stray import/junk in build script | delete the junk import — build scripts need no imports |
| `Kotlin metadata ... expected version` | kotlin plugin older than a dependency's kotlin | align kotlin plugin version to what the library expects, or downgrade the library |

## Known-good constellation (this machine, verified working)
gradle 8.9 wrapper · AGP 8.7.3 · kotlin 2.0.21 · JDK 21 (Android Studio JBR)
· compileSdk 34 or 35 (34 installed locally; 35 auto-downloads) ·
core-ktx 1.15.0 REQUIRES compileSdk 35; use 1.13.1 with compileSdk 34.

## Non-gradle quick table
- pip: `ResolutionImpossible` → loosen one pin at a time, prefer installed versions (`pip list`).
- npm: `ERESOLVE` → align peer majors; `npm ls <pkg>` to see the tree; avoid --force.
- Never fix a dependency error by editing application source, and never fix a
  source error by bumping versions. Classify first.
