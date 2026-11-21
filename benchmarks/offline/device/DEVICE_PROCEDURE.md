# AI-12 device run procedure (Android, adb)

This is how the desktop harness is run on real phones. The desktop numbers say which
models are worth trying; only these device runs say which ones a student's phone can
actually carry. Every number in a device report must name the phone, its SoC and its
physical RAM.

## Target devices

BUILD's floor is **Android 14+, at least 4 GB physical RAM, 64 GB storage**. Most
students use budget phones, so the 4 GB tier decides the default model; higher tiers
unlock larger models. "Virtual RAM" / "RAM expansion" (swap on flash) does **not**
count: record it, but tier by physical RAM (`MemTotal` in `/proc/meminfo`).

| Tier | Physical RAM | Typical phones (examples, not a shortlist) | Models to run |
|---|---|---|---|
| Low (floor) | 4 GB | **Samsung Galaxy A06 (reference, in hand)**; Tecno Spark / Infinix Hot / itel class (MediaTek Helio G or Unisoc) | Qwen3.5 0.8B, Qwen3.5 2B; Gemma 4 E2B only if the desktop peak RSS says it can fit |
| Mid | 6 GB | Infinix Note / Tecno Camon / Galaxy A1x-A2x class | + Gemma 4 E2B (llama.cpp and LiteRT-LM), Gemma 3n E2B, Ministral 3 3B |
| High | 8 GB+ | Galaxy A5x / Pixel / mid-high Snapdragon | + Qwen3.5 4B, Phi-4-mini |

Run every model of a lower tier on the higher tiers too, so each model has a curve
across devices. Also run the Tabwee T80 tablet if it is available, tiered by its
physical RAM.

## Minimum-tier reference device: Samsung Galaxy A06

The first real test phone, and the reference for the 4 GB tier: Samsung Galaxy A06,
4 GB physical RAM, 64 GB storage, connected by USB. Do not trust a marketing page for
anything below: read it from the phone (the SoC is believed to be a MediaTek Helio G85
class chip, **unverified** until read).

```bash
adb shell getprop ro.build.version.release          # Android version (must be 14 or higher)
adb shell getprop ro.build.version.security_patch
adb shell getprop ro.soc.model; adb shell getprop ro.board.platform; adb shell getprop ro.hardware
adb shell "grep -m1 -E 'Hardware|model name' /proc/cpuinfo; grep -c ^processor /proc/cpuinfo"
adb shell "grep -E 'MemTotal|SwapTotal' /proc/meminfo"
```

- **RAM Plus.** The A06 offers "RAM Plus" (4 GB of storage-backed virtual memory). It is
  swap on flash, not RAM. Turn it **off** for every run (Settings > Battery and device
  care > Memory > RAM Plus, restart) and write the state used in `device.txt`; if it
  cannot be turned off, say so and treat peak-RAM results as optimistic. `MemTotal`
  stays the tier, and `SwapTotal` shows whether RAM Plus was really off.
- **USB and battery.** Over USB the phone charges, which hides drain. Pair once over
  USB, then run unplugged: `adb tcpip 5555`, unplug, `adb connect <phone-ip>:5555`
  (phone and PC on the same Wi-Fi). `host_monitor.sh` records `plugged`, and
  `summarise_device.py` reports no battery figure for a run that was charging.
- **Shared phone.** The mobile-ui lane also uses this phone. Peak-RAM, thermal and
  battery runs need it exclusive, with nothing else installed or running, and cool
  (battery `temperature` at or below 330, i.e. 33.0 C). Agree a time window with that
  lane first, and record any app left running.
- **Nothing is installed or run until the owner confirms** the phone is connected and
  USB debugging is on. `adb` itself is not yet on the PC (see below).

### adb: not installed yet, needs approval first

Google's Android SDK Platform-Tools for Windows: file `platform-tools-latest-windows.zip`,
from `https://dl.google.com/android/repository/platform-tools-latest-windows.zip`,
8,188,601 bytes (7.8 MB, last modified 2026-11-19 by the server). It is a zip to extract
into a user folder; there is no installer and nothing is added to the system PATH
unless the owner chooses. Do not download it before the owner says yes.

### No-install-on-the-phone path (preferred on the shared A06)

Termux (below) needs an APK installed on the phone. The alternative leaves nothing but
files in `/data/local/tmp/ai12` (deleted afterwards): cross-compile llama.cpp for
Android arm64 with the NDK on a hosted runner, `adb push` the binaries and one model,
start `llama-server` with `adb shell`, forward its port with `adb forward`, and let the
PC-side harness drive it, reading `/proc/<pid>/status` through adb for memory. The
adb-driven mode of `bench.py` and the Android CI build are **not built yet**; the
Termux path below works today.

## What you need

- A PC with Android platform-tools (`adb`) and this repository checked out.
- On each phone: Developer options and **Wireless debugging** on (Android 11+). Use
  wireless adb so the phone is on battery; over USB it charges and the battery figures
  are meaningless.
- **Termux** from F-Droid or its GitHub releases (the Play Store build is outdated).
  Installing it is the phone owner's decision.
- About 2 GB free storage per model being tested, plus 1 GB for the build.

## Record the device first

```bash
adb shell getprop ro.product.manufacturer
adb shell getprop ro.product.model
adb shell getprop ro.soc.manufacturer
adb shell getprop ro.soc.model
adb shell getprop ro.build.version.release
adb shell cat /proc/meminfo
```

Put the answers in the device's results folder as `device.txt`, plus any advertised
"virtual RAM" setting.

## One-time setup on the phone

1. Copy the harness to shared storage and into Termux:
   ```bash
   adb push benchmarks/offline /sdcard/Download/ai12-harness
   ```
   In Termux: `termux-setup-storage`, then
   `cp -r ~/storage/downloads/ai12-harness ~/ai12-harness`.
2. Build llama.cpp on the phone (about 15-40 minutes on a budget phone):
   ```bash
   bash ~/ai12-harness/device/termux_setup.sh          # CPU
   bash ~/ai12-harness/device/termux_setup.sh vulkan   # plus the Vulkan GPU backend
   bash ~/ai12-harness/device/termux_setup.sh opencl   # plus OpenCL (Adreno GPUs only)
   ```
   Hexagon (Snapdragon NPU) builds need Qualcomm's Hexagon SDK and a cross-compile on
   the PC; they are out of scope for the budget tier, where MediaTek and Unisoc SoCs
   dominate. Record "not run" for them.

## Per model

1. Fetch the weights on the PC (pinned and sha256-checked), then push and copy them
   into Termux's own storage. mmap from `/sdcard` goes through FUSE and is slow, which
   would distort cold start.
   ```bash
   python benchmarks/offline/fetch_model.py qwen3.5-0.8b-q4km --dest ~/ai12-models
   adb push ~/ai12-models/Qwen3.5-0.8B-Q4_K_M.gguf /sdcard/Download/
   ```
   In Termux: `cp ~/storage/downloads/Qwen3.5-0.8B-Q4_K_M.gguf ~/ai12/models/`
2. Prepare the phone: close other apps, screen brightness fixed at about 50%, Wi-Fi on
   (for adb) and mobile data off, battery above 60%, and let it cool until
   `adb shell dumpsys battery` reports `temperature` at or below 330 (33.0 C).
3. On the PC, start the monitor and leave it running:
   ```bash
   bash benchmarks/offline/device/host_monitor.sh runs/<phone>-monitor.csv
   ```
4. In Termux (keep the screen on; Termux's notification "Acquire wakelock" helps):
   ```bash
   bash ~/ai12-harness/device/run_device.sh qwen3.5-0.8b-q4km ~/ai12/models/Qwen3.5-0.8B-Q4_K_M.gguf cpu 4
   ```
   Repeat with `vulkan` or `opencl` where built, and with 2 and 6 threads on one model
   per phone to find the best thread count (budget SoCs often have only 2 big cores).
5. Pull the results and attach the battery and thermal figures:
   ```bash
   adb pull /sdcard/ai12 benchmarks/offline/runs/devices/<phone>
   python benchmarks/offline/device/summarise_device.py \
     benchmarks/offline/runs/devices/<phone>/<model>.cpu.json \
     runs/<phone>-monitor.csv \
     benchmarks/offline/runs/devices/<phone>/<model>.cpu.soak.csv
   ```
6. If the phone kills Termux (the run stops, or `adb logcat -b events | grep am_kill`
   shows `com.termux`), record the model as **does not fit** on that phone at that
   context size. That is a result, not a failure of the procedure.
7. Delete the weights from the phone before the next model.

## What each figure means

| Figure | Where it comes from | Why it matters |
|---|---|---|
| Download MB | file size | students pay for data |
| Peak RSS MB | kernel `VmHWM` of llama-server | must fit beside Android and the app on a 4 GB phone |
| Cold start s | spawn until the model reports loaded | first answer after opening the app |
| TTFT s | request until the first token | how long the student waits |
| Decode tok/s | server timings | reading speed of the answer |
| Sustained ratio | last vs first 10% of a 10-minute soak | thermal throttling |
| Battery %/hour | `dumpsys battery`, wireless adb only | a study session must not drain the phone |
| Max temperature | `dumpsys battery` and `dumpsys thermalservice` | comfort and throttling |
| Min MemAvailable | `/proc/meminfo` | how close Android came to killing apps |

## Google AI Edge (LiteRT-LM) on the phone

Gemma `.litertlm` files run in Google's LiteRT-LM runtime, not llama.cpp, so their
figures form a separate row and are not compared as if the runtimes were identical.
Two ways to measure them:

1. `benchmarks/offline/litert_bench.py` in Termux, if the `litert-lm-api` Android wheel
   installs there (`pip install litert-lm-api==0.19.2`); it reports the same metrics as
   the llama.cpp runs.
2. Otherwise, Google's AI Edge Gallery app, whose model page shows prefill and decode
   speed and time to first token. Record those by hand with the phone details; quality
   tasks then come from the desktop LiteRT-LM run only.

## Then

Regenerate the tables for the devices folder:

```bash
python benchmarks/offline/report.py benchmarks/offline/runs/devices/<phone> \
  --tables benchmarks/offline/runs/devices/<phone>/TABLES.md
```
