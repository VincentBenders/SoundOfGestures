# Project Context: Audiovisual Dance Experience (Single-Stem Low-Pass Filter Demo)

## Overview
We are building an interactive installation in Python where body movement (tracked via MediaPipe) shapes a curated, human-made ambient soundscape[cite: 1]. 
For this initial 2-day sprint, we are decoupling camera tracking from the audio engine[cite: 1]. This repository is the **Single-Stem Filter & Mouse Simulator** prototype.

## Architecture & Data Contract
- **Language:** Python 3.x (`sounddevice`, `scipy.signal` or `pedalboard`, `soundfile`, plus `pygame` or `tkinter` for mouse tracking and UI). Note: Do not use `pygame.mixer` for audio playback, as it lacks real-time low-pass filtering.
- **Normalized Input Contract:** All spatial inputs must be normalized floats between `0.0` and `1.0` (`0.0` = bottom edge, `1.0` = top edge) so we can easily swap mouse coordinates for MediaPipe `LEFT_WRIST` / `RIGHT_WRIST` or body bounding-box scale later[cite: 1].
- **Audio File:** A single pre-recorded ambient loop named `dreamydrone.wav` located in the root workspace directory[cite: 1].

## Core Audio Requirements
1. **Seamless Looping:** Load `dreamydrone.wav` and stream it continuously in an infinite loop without clicks or gaps at the loop point.
2. **Mouse Mapping (Single Parameter — Low-Pass Filter):**
   - **Mouse Y (`0.0` to `1.0`):** Controls the cutoff frequency of a real-time **Low-Pass Filter**[cite: 1].
   - **Logarithmic Scaling:** Map `0.0` (bottom of screen) to `150 Hz` (heavily muffled/underwater) and `1.0` (top of screen) to `8000 Hz` (wide open/bright) using an exponential curve (`min_freq * (max_freq / min_freq) ** smooth_y`) so the frequency sweep sounds natural to human ears.
3. **Jitter Smoothing:** Apply an exponential moving average (`smooth_y = (smooth_y * 0.85) + (raw_y * 0.15)`) to the incoming Y coordinate so rapid movements never cause digital filter clicks or stepping[cite: 1].
4. **Visual Debug UI:** Display a clean window showing the raw normalized Y value, the smoothed Y value, the current cutoff frequency in Hz, and a visual bar representing the filter opening.

## Team Workflow Constraints
- Keep the audio processing callback cleanly separated from the mouse input loop (e.g., an `update_filter(norm_y)` function) so my teammate can easily plug in MediaPipe coordinates tomorrow[cite: 1].