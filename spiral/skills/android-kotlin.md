---
name: android-kotlin
description: Android app work in Kotlin — gradle, activities, layouts, resources, viewBinding, RecyclerView, TTS, MediaPlayer, animations, manifest. Use for any .kt, .kts, AndroidManifest, or res/ task.
---
# Android/Kotlin discipline

## Cross-file coherence (the #1 autonomous-agent failure)
- Reference ONLY ids/classes/resources that exist RIGHT NOW in the repo, or that
  this same task creates. Never reference something "a later task will add".
- One data model per concept, ever. Before creating a class, grep for an existing
  one (`grep -r "data class Message"`). Extend; don't duplicate.
- Every `findViewById(R.id.x)` must match an `android:id="@+id/x"` in the exact
  layout this activity inflates. Check the layout file first.
- Every Activity you `startActivity(...)` to must exist AND be declared in
  AndroidManifest.xml inside `<application>`.
- Drawable references (`@drawable/name`) resolve to res/drawable/name.xml —
  the FILENAME, not an android:name attribute. Match exactly.

## Kotlin/view patterns
- Bind views before use; `lateinit var` + use-before-init = runtime crash.
- Prefer viewBinding (enabled in this project) over findViewById:
  `private lateinit var b: ActivityMainBinding` →
  `b = ActivityMainBinding.inflate(layoutInflater); setContentView(b.root)` →
  `b.etMessage.text`. Binding class name = layout file name in PascalCase.
- RecyclerView needs: adapter + `layoutManager = LinearLayoutManager(this)`.
  After adding an item: `adapter.notifyItemInserted(n)` and `recycler.scrollToPosition(n)`.
- Color strings must be `#AARRGGBB` or `#RRGGBB` — `Color.parseColor("E6FFFFFF")`
  without `#` throws at RUNTIME (compiles fine). Audit every parseColor.
- Handler loops: `Handler(Looper.getMainLooper()).postDelayed({...}, ms)`; cancel
  in `onDestroy` (`removeCallbacksAndMessages(null)`), or leak + crash.

## Sound & speech (sirens, announcements)
- TTS: `TextToSpeech(this) { status -> if (status == TextToSpeech.SUCCESS) tts.language = Locale.US }`
  then `tts.speak(text, TextToSpeech.QUEUE_ADD, null, "id")`. Shut down in onDestroy.
  MAX VOLUME: set stream volume via AudioManager before speaking:
  `audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC), 0)`.
- Siren without an audio asset: `ToneGenerator(AudioManager.STREAM_MUSIC, 100)` +
  `startTone(ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK, 2000)` — no res/raw file
  needed. If a res/raw file is used, it must physically exist before referencing.
- MediaPlayer: `create(this, R.raw.x)` may return null; guard it; `release()` after.

## Animations
- Simple + reliable: `ValueAnimator.ofFloat/ofInt` + updateListener + `invalidate()`.
- `ObjectAnimator.ofFloat(view, "translationY", 0f, 100f)` for view properties.
- Progress feel: animate a ProgressBar's progress with ValueAnimator over ~2-4s.
- Custom View: draw in `onDraw(canvas)` only; state changes → `invalidate()`;
  never allocate Paint inside onDraw.

## Gradle hygiene
- Build scripts import nothing exotic: a plain `plugins { }` block. Stray
  `import org.jetbrains...` lines at the top of .kts files break configuration.
- New permissions (VIBRATE, etc.) go in the manifest, not gradle.
- Don't bump dependency or SDK versions to fix code errors — that's a
  dependency-medic decision.
