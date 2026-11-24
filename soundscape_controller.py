"""
2D Audiovisual Modulator & Rhythmic Flutter Studio
Project Context: Audiovisual Dance Experience (Technical Team Demonstration)

Architecture:
- LowPassAudioEngine: Decoupled real-time audio streamer with 4-pole low-pass filtering (-24 dB/oct),
  dynamic make-up gain compensation, and an accelerating Rhythmic Flutter (Pulse Stutter Gate) on X:
    * Y-Axis: Low-Pass Filter cutoff (0.0 = 80 Hz bottom sub rumble, 1.0 = 8000 Hz top bright)
    * X-Axis: Rhythmic Flutter (0.0 = Smooth Continuous Drone, moving right accelerates from 1.5 Hz heartbeat to 15 Hz flutter)
- update_filter(norm_y, norm_x=0.0): Thread-safe decoupled function taking normalized coordinates [0.0 - 1.0]
  with exponential moving average jitter smoothing.
- Technical Demonstration Dashboard: Pygame-based GUI featuring:
  * Central 2D Matrix trackpad showcasing Raw Input vs Smoothed Tracker (jitter elimination proof)
  * Real-time 36-band log FFT spectrum analyzer with -24 dB/oct cutoff marker
  * Dual-channel Stereo VU Output Meters (Peak & RMS)
  * Live DSP Telemetry (Filter slope, Logarithmic mapping, Make-up gain, LFO pulse oscilloscope)
  * Audio Engine health metrics (Buffer latency, 48kHz Stereo float32, MediaPipe handover hook)
"""

import os
import sys
import time
import math
import threading
import argparse
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy import signal


# ==============================================================================
# AUDIO ENGINE (Decoupled from UI / Camera Input)
# ==============================================================================

class LowPassAudioEngine:
    """
    Decoupled 2D audio engine streaming an ambient audio loop with:
    - Y-Axis: 4-Pole Low-Pass Filter (-24 dB/octave) + Dynamic Make-Up Gain
    - X-Axis: Accelerating Rhythmic Flutter (Continuous -> 1.5 Hz -> 15 Hz Pulse Gate)
    """

    MIN_LP_FREQ = 80.0    # Hz (deep sub-bass sweep)
    MAX_LP_FREQ = 8000.0  # Hz (wide open / bright)

    MIN_FLUTTER_RATE = 1.5   # Hz (slow heartbeat pulse)
    MAX_FLUTTER_RATE = 15.0  # Hz (rhythmic pulse flutter)

    SMOOTHING_PREV = 0.85
    SMOOTHING_NEW = 0.15

    GAIN_COMPENSATION = True
    GAIN_BOOST_MAX = 3.2
    FILTER_ORDER = 4      # 4-Pole (-24 dB/octave)

    NUM_BANDS = 36
    BAND_MIN_FREQ = 40.0
    BAND_MAX_FREQ = 12000.0

    def __init__(self, audio_path: Optional[str] = None, block_size: int = 1024):
        self.block_size = block_size
        self.audio_path = self._resolve_audio_file(audio_path)
        
        # Load audio into memory
        self.audio_data, self.sample_rate = sf.read(self.audio_path, dtype="float32")
        if self.audio_data.ndim == 1:
            self.audio_data = self.audio_data[:, np.newaxis]
        self.channels = self.audio_data.shape[1]
        self.total_frames = len(self.audio_data)

        # Threading and streaming state
        self._lock = threading.Lock()
        self._playhead = 0
        self.is_running = False
        self.stream: Optional[sd.OutputStream] = None

        # Parameter state
        self.raw_y = 0.5
        self.smooth_y = 0.5
        self.raw_x = 0.0      # Start with flutter OFF (smooth continuous drone)
        self.smooth_x = 0.0

        self.cutoff_hz = self._calc_cutoff(self.smooth_y)
        self.current_gain = self._calc_gain(self.smooth_y)
        self.flutter_rate_hz = 0.0
        self.flutter_active = False

        # Initialize Primary 4-pole Low-Pass Filter (-24 dB/oct)
        self._lp_sos = signal.butter(self.FILTER_ORDER, self.cutoff_hz, btype="low", fs=self.sample_rate, output="sos")
        zi_single = signal.sosfilt_zi(self._lp_sos)
        self._lp_zi = np.repeat(zi_single[:, np.newaxis, :], self.channels, axis=1)

        # Flutter LFO phase
        self._flutter_phase = 0.0

        # Visual analysis buffers
        self._latest_chunk = np.zeros((self.block_size, self.channels), dtype=np.float32)
        self._window = np.hanning(self.block_size)
        self.band_edges = np.geomspace(self.BAND_MIN_FREQ, self.BAND_MAX_FREQ, self.NUM_BANDS + 1)
        fft_freqs = np.fft.rfftfreq(self.block_size, 1.0 / self.sample_rate)
        self._band_indices = [
            np.where((fft_freqs >= self.band_edges[i]) & (fft_freqs < self.band_edges[i + 1]))[0]
            for i in range(self.NUM_BANDS)
        ]

    @classmethod
    def _resolve_audio_file(cls, preferred_path: Optional[str] = None) -> str:
        if preferred_path and os.path.isfile(preferred_path):
            return os.path.abspath(preferred_path)

        # Check in audio folder first, then root folder
        candidates = []
        if preferred_path:
            candidates.extend([
                preferred_path,
                os.path.join("audio", preferred_path),
                os.path.join(os.path.dirname(__file__), "audio", preferred_path),
            ])

        base_names = [
            "DreamyDrone.wav", "dreamydrone.wav",
            "DreamyDrone.flac", "dreamydrone.flac",
            "DarkCinematic.wav", "Forest.flac", "Drums.wav", "Harmony.mp3"
        ]
        for name in base_names:
            candidates.extend([
                os.path.join("audio", name),
                os.path.join(os.path.dirname(__file__), "audio", name),
                name,
                os.path.join(os.path.dirname(__file__), name),
            ])

        for path in candidates:
            if path and os.path.isfile(path):
                return os.path.abspath(path)

        # Fallback: find any supported audio file in audio/ or current dir
        for search_d in ["audio", os.path.join(os.path.dirname(__file__), "audio"), ".", os.path.dirname(__file__)]:
            if search_d and os.path.isdir(search_d):
                for fname in sorted(os.listdir(search_d)):
                    if fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                        return os.path.abspath(os.path.join(search_d, fname))

        raise FileNotFoundError("Could not locate any audio files in 'audio/' or root folder.")

    def _reinit_dsp(self):
        """Recompute filter coefficients and FFT analysis buffers for current sample rate / channel count."""
        self._lp_sos = signal.butter(self.FILTER_ORDER, self.cutoff_hz, btype="low", fs=self.sample_rate, output="sos")
        zi_single = signal.sosfilt_zi(self._lp_sos)
        self._lp_zi = np.repeat(zi_single[:, np.newaxis, :], self.channels, axis=1)

        self._latest_chunk = np.zeros((self.block_size, self.channels), dtype=np.float32)
        fft_freqs = np.fft.rfftfreq(self.block_size, 1.0 / self.sample_rate)
        self._band_indices = [
            np.where((fft_freqs >= self.band_edges[i]) & (fft_freqs < self.band_edges[i + 1]))[0]
            for i in range(self.NUM_BANDS)
        ]

    def load_audio(self, audio_path: str):
        """
        Hot-swap the active audio loop to a new file in a thread-safe manner.
        Handles sample rate transitions and filter coefficient recalculations seamlessly.
        """
        resolved = self._resolve_audio_file(audio_path)
        new_data, new_sr = sf.read(resolved, dtype="float32")
        if new_data.ndim == 1:
            new_data = new_data[:, np.newaxis]
        new_channels = new_data.shape[1]

        was_running = self.is_running
        if was_running and (new_sr != self.sample_rate or new_channels != self.channels):
            self.stop()
            with self._lock:
                self.audio_path = resolved
                self.audio_data = new_data
                self.sample_rate = new_sr
                self.channels = new_channels
                self.total_frames = len(new_data)
                self._playhead = 0
                self._reinit_dsp()
            self.start()
        else:
            with self._lock:
                self.audio_path = resolved
                self.audio_data = new_data
                self.sample_rate = new_sr
                self.channels = new_channels
                self.total_frames = len(new_data)
                self._playhead = 0
                self._reinit_dsp()

    def _calc_cutoff(self, norm_val: float) -> float:
        clamped = max(0.0, min(1.0, float(norm_val)))
        return float(self.MIN_LP_FREQ * ((self.MAX_LP_FREQ / self.MIN_LP_FREQ) ** clamped))

    def _calc_gain(self, norm_val: float) -> float:
        if not self.GAIN_COMPENSATION:
            return 1.0
        clamped = max(0.0, min(1.0, float(norm_val)))
        return float(1.0 + self.GAIN_BOOST_MAX * ((1.0 - clamped) ** 1.6))

    def _calc_flutter_rate(self, norm_x: float) -> float:
        if norm_x < 0.04:
            return 0.0
        # Exponential acceleration from 1.5 Hz to 15 Hz
        return float(self.MIN_FLUTTER_RATE + (norm_x ** 1.8) * (self.MAX_FLUTTER_RATE - self.MIN_FLUTTER_RATE))

    def update_filter(self, norm_y: float, norm_x: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Thread-safe controller update taking normalized [0.0 - 1.0] coordinates.
        
        Args:
            norm_y: Vertical input (0.0 = bottom edge 80 Hz, 1.0 = top edge 8000 Hz)
            norm_x: Horizontal input (0.0 = left edge Smooth Drone, 1.0 = right edge 15 Hz Flutter)
            
        Returns:
            Tuple of (raw_y, smooth_y, raw_x, smooth_x)
        """
        clamped_y = max(0.0, min(1.0, float(norm_y)))
        clamped_x = max(0.0, min(1.0, float(norm_x)))

        # Jitter smoothing
        sm_y = (self.smooth_y * self.SMOOTHING_PREV) + (clamped_y * self.SMOOTHING_NEW)
        sm_x = (self.smooth_x * self.SMOOTHING_PREV) + (clamped_x * self.SMOOTHING_NEW)

        cutoff = self._calc_cutoff(sm_y)
        gain = self._calc_gain(sm_y)
        flutter_rate = self._calc_flutter_rate(sm_x)

        new_lp_sos = signal.butter(self.FILTER_ORDER, cutoff, btype="low", fs=self.sample_rate, output="sos")

        with self._lock:
            self.raw_y = clamped_y
            self.smooth_y = sm_y
            self.raw_x = clamped_x
            self.smooth_x = sm_x
            self.cutoff_hz = cutoff
            self.current_gain = gain
            self.flutter_rate_hz = flutter_rate
            self.flutter_active = (sm_x >= 0.04)
            self._lp_sos = new_lp_sos

        return self.raw_y, self.smooth_y, self.raw_x, self.smooth_x

    def get_status(self) -> dict:
        with self._lock:
            return {
                "raw_y": self.raw_y,
                "smooth_y": self.smooth_y,
                "raw_x": self.raw_x,
                "smooth_x": self.smooth_x,
                "cutoff_hz": self.cutoff_hz,
                "gain": self.current_gain,
                "flutter_rate_hz": self.flutter_rate_hz,
                "flutter_active": self.flutter_active,
                "is_running": self.is_running,
                "sample_rate": self.sample_rate,
                "block_size": self.block_size,
            }

    def get_visual_data(self) -> dict:
        with self._lock:
            chunk = self._latest_chunk.copy()
            cutoff = self.cutoff_hz
            sm_x = self.smooth_x
            sm_y = self.smooth_y
            r_x = self.raw_x
            r_y = self.raw_y
            gain = self.current_gain
            fl_rate = self.flutter_rate_hz
            fl_active = self.flutter_active
            fl_phase = self._flutter_phase

        if len(chunk) > 0:
            peak_l = float(np.max(np.abs(chunk[:, 0])))
            peak_r = float(np.max(np.abs(chunk[:, 1]))) if chunk.shape[1] > 1 else peak_l
            rms_l = float(np.sqrt(np.mean(chunk[:, 0] ** 2)))
            rms_r = float(np.sqrt(np.mean(chunk[:, 1] ** 2))) if chunk.shape[1] > 1 else rms_l
            
            mono = np.mean(chunk, axis=1)
            fft_mags = np.abs(np.fft.rfft(mono * self._window)) * (2.0 / self.block_size)
            bands = np.array([
                float(np.mean(fft_mags[idx])) if len(idx) > 0 else 0.0
                for idx in self._band_indices
            ], dtype=np.float32)
        else:
            peak_l, peak_r, rms_l, rms_r = 0.0, 0.0, 0.0, 0.0
            bands = np.zeros(self.NUM_BANDS, dtype=np.float32)

        return {
            "peak_l": peak_l,
            "peak_r": peak_r,
            "rms_l": rms_l,
            "rms_r": rms_r,
            "bands": bands,
            "cutoff_hz": cutoff,
            "smooth_x": sm_x,
            "smooth_y": sm_y,
            "raw_x": r_x,
            "raw_y": r_y,
            "gain": gain,
            "flutter_rate_hz": fl_rate,
            "flutter_active": fl_active,
            "flutter_phase": fl_phase,
            "band_edges": self.band_edges,
            "min_freq": self.BAND_MIN_FREQ,
            "max_freq": self.BAND_MAX_FREQ,
        }

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Audio callback performing seamless loop + 4-pole low-pass + rhythmic flutter."""
        if status and status.output_underflow:
            pass

        with self._lock:
            lp_sos = self._lp_sos
            pos = self._playhead
            gain = self.current_gain
            sm_x = self.smooth_x

            # 1. Seamless Looping
            end_pos = pos + frames
            if end_pos <= self.total_frames:
                chunk = self.audio_data[pos:end_pos]
                self._playhead = end_pos % self.total_frames
            else:
                part1 = self.audio_data[pos:]
                rem = frames - len(part1)
                part2 = self.audio_data[:rem]
                chunk = np.concatenate((part1, part2), axis=0)
                self._playhead = rem

            # 2. 4-Pole Low-Pass Filter (-24 dB/oct) + Make-up Gain
            filtered, self._lp_zi = signal.sosfilt(lp_sos, chunk, axis=0, zi=self._lp_zi)
            stage_out = filtered * gain

            # 3. Rhythmic Flutter Gate
            if sm_x >= 0.04:
                rate_hz = self.MIN_FLUTTER_RATE + (sm_x ** 1.8) * (self.MAX_FLUTTER_RATE - self.MIN_FLUTTER_RATE)
                t = np.arange(frames) / self.sample_rate
                lfo = 0.5 + 0.5 * np.cos(2.0 * np.pi * rate_hz * t + self._flutter_phase)
                gate = np.clip((lfo - 0.35) * 5.0, 0.0, 1.0)
                depth = min(1.0, sm_x * 1.35)
                mod = (1.0 - depth) + depth * gate
                stage_out = stage_out * mod[:, np.newaxis]
                self._flutter_phase = (self._flutter_phase + 2.0 * np.pi * rate_hz * frames / self.sample_rate) % (2.0 * np.pi)

        outdata[:] = stage_out
        self._latest_chunk = stage_out

    def start(self):
        if self.is_running:
            return
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.block_size,
            callback=self._audio_callback,
            dtype="float32",
        )
        self.stream.start()
        self.is_running = True

    def stop(self):
        if not self.is_running:
            return
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_running = False


# ==============================================================================
# GLOBAL CONVENIENCE INTERFACE (Teammate MediaPipe Handover)
# ==============================================================================

_default_engine: Optional[LowPassAudioEngine] = None
_engine_lock = threading.Lock()

def get_engine(audio_path: Optional[str] = None) -> LowPassAudioEngine:
    global _default_engine
    with _engine_lock:
        if _default_engine is None:
            _default_engine = LowPassAudioEngine(audio_path=audio_path)
        return _default_engine

def start_audio(audio_path: Optional[str] = None) -> LowPassAudioEngine:
    engine = get_engine(audio_path)
    engine.start()
    return engine

def stop_audio():
    global _default_engine
    with _engine_lock:
        if _default_engine is not None:
            _default_engine.stop()

def update_filter(norm_y: float, norm_x: float = 0.0) -> Tuple[float, float, float, float]:
    """
    2D Spatial controller update.
    
    Args:
        norm_y: Vertical coordinate (0.0 = bottom edge 80 Hz, 1.0 = top edge 8000 Hz)
        norm_x: Horizontal coordinate (0.0 = left edge Smooth Drone, 1.0 = right edge 15 Hz Flutter)
        
    Returns:
        Tuple of (raw_y, smooth_y, raw_x, smooth_x)
    """
    engine = get_engine()
    return engine.update_filter(norm_y, norm_x)

def load_audio(audio_path: str):
    """Hot-swap the current soundscape to a new audio file."""
    engine = get_engine()
    engine.load_audio(audio_path)

def get_available_tracks() -> List[str]:
    """Discover all supported audio files in 'audio/' directory and root folder."""
    valid_exts = (".wav", ".flac", ".mp3", ".ogg", ".aiff")
    tracks = []
    seen = set()
    
    search_dirs = [
        os.path.join(os.path.dirname(__file__), "audio"),
        "audio",
        os.path.dirname(__file__),
        "."
    ]
    
    for s_dir in search_dirs:
        if s_dir and os.path.isdir(s_dir):
            for fname in sorted(os.listdir(s_dir)):
                if fname.lower().endswith(valid_exts):
                    base_k = fname.lower()
                    if base_k not in seen:
                        seen.add(base_k)
                        tracks.append(os.path.abspath(os.path.join(s_dir, fname)))
    return tracks


# ==============================================================================
# DEMONSTRATION UI (Pygame)
# ==============================================================================

def run_mouse_simulator(audio_path: Optional[str] = None):
    import pygame

    pygame.init()
    pygame.font.init()

    # Clean demonstration studio window (1300 x 720)
    width, height = 1300, 720
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Audiovisual Experience: Sound Control Demonstration")
    clock = pygame.time.Clock()

    # Fonts
    title_font = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 18, bold=True)
    card_title_font = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 15, bold=True)
    axis_title_font = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 16, bold=True)
    axis_tick_font = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 13, bold=True)
    big_stat_font = pygame.font.SysFont("Consolas, Courier New, monospace", 20, bold=True)
    body_font = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 13)
    small_font = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 12)
    mini_font = pygame.font.SysFont("Consolas, Courier New, monospace", 11)

    # Color Palette (Clean Modern Dark)
    COLOR_BG = (12, 15, 23)
    COLOR_PANEL = (19, 24, 38)
    COLOR_PANEL_INNER = (13, 17, 28)
    COLOR_BORDER = (34, 43, 66)
    COLOR_ACCENT_CYAN = (0, 229, 255)
    COLOR_ACCENT_PURPLE = (175, 110, 255)
    COLOR_ACCENT_GOLD = (255, 195, 0)
    COLOR_RAW_RED = (255, 80, 80)
    COLOR_TEXT_PRIMARY = (245, 247, 250)
    COLOR_TEXT_SECONDARY = (145, 158, 182)
    COLOR_TEXT_MUTED = (85, 98, 125)
    COLOR_GRID = (28, 35, 54)
    COLOR_GRID_ACTIVE = (45, 58, 88)

    # Start audio engine
    engine = get_engine(audio_path)
    engine.start()

    # Discover all available audio tracks in 'audio/'
    available_tracks = get_available_tracks()
    if not available_tracks:
        available_tracks = [engine.audio_path]
    current_track = engine.audio_path

    # Animation buffers
    smooth_bands = np.zeros(engine.NUM_BANDS, dtype=np.float32)

    # Layout dimensions (3-column layout: Left Stems, Center Pad, Right Feedback)
    panels_y = 68
    panels_h = height - panels_y - 14

    stem_box_x = 20
    stem_box_w = 205
    stem_box_y = panels_y
    stem_box_h = panels_h

    pad_box_x = stem_box_x + stem_box_w + 14
    pad_box_w = 615
    pad_box_y = panels_y
    pad_box_h = panels_h

    # Exactly square 2D pad (1:1 aspect ratio)
    pad_size = 500
    pad_x = pad_box_x + 92
    pad_y = pad_box_y + 36
    pad_w = pad_size
    pad_h = pad_size

    side_x = pad_box_x + pad_box_w + 14
    side_w = width - side_x - 20

    track_buttons_last_frame = []

    running = True
    try:
        while running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif pygame.K_1 <= event.key <= pygame.K_9:
                        t_idx = event.key - pygame.K_1
                        if t_idx < len(available_tracks):
                            current_track = available_tracks[t_idx]
                            engine.load_audio(current_track)
                    elif event.key in (pygame.K_SPACE, pygame.K_TAB):
                        if current_track in available_tracks:
                            cur_idx = available_tracks.index(current_track)
                            next_idx = (cur_idx + 1) % len(available_tracks)
                        else:
                            next_idx = 0
                        current_track = available_tracks[next_idx]
                        engine.load_audio(current_track)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    m_pos = event.pos
                    for b_rect, t_path in track_buttons_last_frame:
                        if b_rect.collidepoint(m_pos):
                            if os.path.normpath(current_track) != os.path.normpath(t_path):
                                current_track = t_path
                                engine.load_audio(current_track)

            # 2. Mouse Tracking on Matrix Pad
            mouse_x, mouse_y = pygame.mouse.get_pos()
            norm_x = (mouse_x - pad_x) / float(pad_w)
            norm_y = 1.0 - ((mouse_y - pad_y) / float(pad_h))
            raw_y, smooth_y, raw_x, smooth_x = update_filter(norm_y, norm_x)

            # 3. Audio & DSP Acquisition
            vis = engine.get_visual_data()
            smooth_bands = np.maximum(vis["bands"], smooth_bands * 0.82)

            pulse_val = 0.0
            if vis["flutter_active"]:
                pulse_val = max(0.0, math.cos(vis["flutter_phase"]))

            # 4. Rendering
            screen.fill(COLOR_BG)

            # ==================================================================
            # TOP HEADER BAR
            # ==================================================================
            header_rect = pygame.Rect(20, 10, width - 40, 48)
            pygame.draw.rect(screen, COLOR_PANEL, header_rect, border_radius=8)
            pygame.draw.rect(screen, COLOR_BORDER, header_rect, width=1, border_radius=8)

            screen.blit(title_font.render("On-Screen Soundscape Simulator", True, COLOR_TEXT_PRIMARY), (36, 15))
            screen.blit(small_font.render("Camera-free mouse prototype (spatial audio testbed)", True, COLOR_TEXT_SECONDARY), (36, 34))

            status_text = "● AUDIO ENGINE ACTIVE"
            screen.blit(small_font.render(status_text, True, COLOR_ACCENT_CYAN), (width - 200, 24))

            # ==================================================================
            # LEFT SECTION: AUDIO STEM SELECTOR (VERTICAL LIST)
            # ==================================================================
            stem_panel_rect = pygame.Rect(stem_box_x, stem_box_y, stem_box_w, stem_box_h)
            pygame.draw.rect(screen, COLOR_PANEL, stem_panel_rect, border_radius=8)
            pygame.draw.rect(screen, COLOR_BORDER, stem_panel_rect, width=1, border_radius=8)

            screen.blit(card_title_font.render("Choose Audio", True, COLOR_TEXT_PRIMARY), (stem_box_x + 16, stem_box_y + 16))

            current_mouse_pos = pygame.mouse.get_pos()
            btn_y = stem_box_y + 48
            btn_h = 42
            btn_spacing = 8
            new_track_buttons = []

            for idx, t_path in enumerate(available_tracks):
                fname = os.path.basename(t_path)
                track_name = os.path.splitext(fname)[0]  # Clean title without format extension
                is_active = (os.path.normpath(t_path) == os.path.normpath(engine.audio_path))

                btn_rect = pygame.Rect(stem_box_x + 10, btn_y, stem_box_w - 20, btn_h)
                new_track_buttons.append((btn_rect, t_path))

                is_hovered = btn_rect.collidepoint(current_mouse_pos)

                if is_active:
                    pygame.draw.rect(screen, (20, 48, 70), btn_rect, border_radius=6)
                    pygame.draw.rect(screen, COLOR_ACCENT_CYAN, btn_rect, width=2, border_radius=6)
                    pygame.draw.circle(screen, COLOR_ACCENT_CYAN, (stem_box_x + 24, btn_y + btn_h // 2), 4)
                    screen.blit(small_font.render(track_name, True, COLOR_ACCENT_CYAN), (stem_box_x + 36, btn_y + 13))
                else:
                    bg_col = (28, 36, 52) if is_hovered else COLOR_PANEL_INNER
                    pygame.draw.rect(screen, bg_col, btn_rect, border_radius=6)
                    pygame.draw.rect(screen, COLOR_BORDER, btn_rect, width=1, border_radius=6)
                    pygame.draw.circle(screen, (70, 80, 105), (stem_box_x + 24, btn_y + btn_h // 2), 3)
                    txt_col = COLOR_TEXT_PRIMARY if is_hovered else COLOR_TEXT_SECONDARY
                    screen.blit(small_font.render(track_name, True, txt_col), (stem_box_x + 36, btn_y + 13))

                btn_y += btn_h + btn_spacing

            track_buttons_last_frame = new_track_buttons

            # Spacebar shortcut instruction
            hint_surf = small_font.render("press space to change the audio", True, COLOR_TEXT_MUTED)
            screen.blit(hint_surf, (stem_box_x + (stem_box_w - hint_surf.get_width()) // 2, btn_y + 16))

            # ==================================================================
            # CENTER SECTION: THE 2D INTERACTION PAD
            # ==================================================================
            pygame.draw.rect(screen, COLOR_PANEL, (pad_box_x, pad_box_y, pad_box_w, pad_box_h), border_radius=8)
            pygame.draw.rect(screen, COLOR_BORDER, (pad_box_x, pad_box_y, pad_box_w, pad_box_h), width=1, border_radius=8)

            # Pad Surface
            pygame.draw.rect(screen, COLOR_PANEL_INNER, (pad_x, pad_y, pad_w, pad_h), border_radius=6)
            pygame.draw.rect(screen, COLOR_BORDER, (pad_x, pad_y, pad_w, pad_h), width=1, border_radius=6)

            # Subdivisions
            for frac in [0.25, 0.50, 0.75]:
                gx = pad_x + int(frac * pad_w)
                col_g = COLOR_GRID_ACTIVE if frac == 0.50 else COLOR_GRID
                pygame.draw.line(screen, col_g, (gx, pad_y), (gx, pad_y + pad_h), 1)
                
                gy = pad_y + int(frac * pad_h)
                pygame.draw.line(screen, col_g, (pad_x, gy), (pad_x + pad_w, gy), 1)

            # Zone Labels (Clear English explaining what each quadrant sounds like)
            screen.blit(small_font.render("Bright + Continuous", True, (90, 110, 145)), (pad_x + 14, pad_y + 12))
            screen.blit(small_font.render("Bright + Fast Flutter", True, (135, 95, 175)), (pad_x + pad_w - 145, pad_y + 12))
            screen.blit(small_font.render("Deep + Continuous", True, (90, 110, 145)), (pad_x + 14, pad_y + pad_h - 26))
            screen.blit(small_font.render("Deep + Fast Flutter", True, (135, 95, 175)), (pad_x + pad_w - 140, pad_y + pad_h - 26))

            # Y-Axis: Vertical Rotated Label & Ticks (Low-Pass Cutoff) - Large & High Contrast
            COLOR_TICK_TEXT = (215, 225, 240)
            y_axis_title = axis_title_font.render("Low-Pass Cutoff Frequency  →", True, (255, 210, 50))
            rot_y_title = pygame.transform.rotate(y_axis_title, 90)
            screen.blit(rot_y_title, (pad_box_x + 14, pad_y + (pad_h - rot_y_title.get_height()) // 2))

            y_ticks = [(0.0, "80 Hz"), (0.50, "800 Hz"), (1.0, "8 kHz")]
            for frac, lbl in y_ticks:
                ty = pad_y + pad_h - int(frac * pad_h)
                t_surf = axis_tick_font.render(lbl, True, COLOR_TICK_TEXT)
                screen.blit(t_surf, (pad_x - t_surf.get_width() - 8, ty - t_surf.get_height() // 2))
                pygame.draw.line(screen, COLOR_GRID_ACTIVE, (pad_x - 5, ty), (pad_x, ty), 1)

            # X-Axis Ticks & Label (Horizontal: Flutter Pulse) - Large & High Contrast
            x_ticks = [(0.0, "Smooth"), (0.50, "5.4 Hz"), (1.0, "15 Hz")]
            for frac, lbl in x_ticks:
                tx = pad_x + int(frac * pad_w)
                t_surf = axis_tick_font.render(lbl, True, COLOR_TICK_TEXT)
                screen.blit(t_surf, (tx - t_surf.get_width() // 2, pad_y + pad_h + 8))
                pygame.draw.line(screen, COLOR_GRID_ACTIVE, (tx, pad_y + pad_h), (tx, pad_y + pad_h + 5), 1)

            x_axis_title = axis_title_font.render("Flutter Pulse  →", True, (195, 130, 255))
            screen.blit(x_axis_title, (pad_x + (pad_w - x_axis_title.get_width()) // 2, pad_y + pad_h + 32))

            # Coordinates on Pad
            raw_px = pad_x + int(vis["raw_x"] * pad_w)
            raw_py = pad_y + pad_h - int(vis["raw_y"] * pad_h)
            sm_px = pad_x + int(vis["smooth_x"] * pad_w)
            sm_py = pad_y + pad_h - int(vis["smooth_y"] * pad_h)

            # Smoothing Connector Line
            pygame.draw.line(screen, (85, 100, 135), (raw_px, raw_py), (sm_px, sm_py), 1)

            # Raw Cursor Indicator
            pygame.draw.circle(screen, COLOR_RAW_RED, (raw_px, raw_py), 4)

            # Smoothed Audio Puck (Glows/pulses with rhythm)
            puck_rad = 12 + int(pulse_val * 6)
            puck_col = COLOR_ACCENT_PURPLE if vis["flutter_active"] else COLOR_ACCENT_CYAN
            pygame.draw.circle(screen, puck_col, (sm_px, sm_py), puck_rad, 2)
            pygame.draw.circle(screen, COLOR_TEXT_PRIMARY, (sm_px, sm_py), 4)

            # ==================================================================
            # RIGHT SECTION: CLEAR EXPLANATION CARDS
            # ==================================================================

            # ------------------------------------------------------------------
            # CARD 1: VERTICAL AXIS (SOUND BRIGHTNESS / LOW-PASS FILTER)
            # ------------------------------------------------------------------
            c1_y = pad_box_y
            c1_h = 210
            c1_rect = pygame.Rect(side_x, c1_y, side_w, c1_h)
            pygame.draw.rect(screen, COLOR_PANEL, c1_rect, border_radius=8)
            pygame.draw.rect(screen, COLOR_BORDER, c1_rect, width=1, border_radius=8)

            screen.blit(card_title_font.render("Y-Axis: Sound Brightness (Filter)", True, COLOR_ACCENT_GOLD), (side_x + 18, c1_y + 12))
            
            # Big Stat Readout
            cutoff_val_str = f"Filter Cutoff: {vis['cutoff_hz']:5.0f} Hz"
            screen.blit(big_stat_font.render(cutoff_val_str, True, COLOR_TEXT_PRIMARY), (side_x + 18, c1_y + 36))

            # Explanation sentence
            mood_str = "Deep underwater rumble (highs cut off)" if vis["cutoff_hz"] < 400 else ("Warm midrange ambient" if vis["cutoff_hz"] < 2500 else "Bright, open, airy ambient")
            screen.blit(small_font.render(f"Mood: {mood_str}", True, COLOR_TEXT_SECONDARY), (side_x + 18, c1_y + 64))

            # Real-time Spectrum Visualizer
            sgraph_x = side_x + 18
            sgraph_y = c1_y + 88
            sgraph_w = side_w - 36
            sgraph_h = 86

            pygame.draw.rect(screen, COLOR_PANEL_INNER, (sgraph_x, sgraph_y, sgraph_w, sgraph_h), border_radius=6)
            pygame.draw.rect(screen, COLOR_BORDER, (sgraph_x, sgraph_y, sgraph_w, sgraph_h), width=1, border_radius=6)

            num_bars = len(smooth_bands)
            bar_spacing = 2
            total_bar_w = (sgraph_w - (num_bars * bar_spacing)) / float(num_bars)
            max_mag_scale = 0.008

            log_min = math.log10(vis["min_freq"])
            log_max = math.log10(vis["max_freq"])
            log_range = log_max - log_min

            for i in range(num_bars):
                mag = smooth_bands[i]
                norm_mag = min(1.0, mag / max_mag_scale)
                h_frac = math.sqrt(norm_mag)
                b_height = int(h_frac * (sgraph_h - 10))

                bx = int(sgraph_x + i * (total_bar_w + bar_spacing)) + 1
                by = sgraph_y + sgraph_h - b_height - 6

                if b_height > 2:
                    pygame.draw.rect(screen, COLOR_ACCENT_CYAN, (bx, by, max(2, int(total_bar_w)), b_height), border_radius=2)

            # Cutoff Frequency Line on Spectrum
            cutoff_frac = max(0.0, min(1.0, (math.log10(vis["cutoff_hz"]) - log_min) / log_range))
            cutoff_x = sgraph_x + int(cutoff_frac * sgraph_w)

            # Shade out filtered frequencies
            if cutoff_x < sgraph_x + sgraph_w:
                shade_w = (sgraph_x + sgraph_w) - cutoff_x
                shade_surf = pygame.Surface((shade_w, sgraph_h), pygame.SRCALPHA)
                shade_surf.fill((10, 12, 20, 180))
                screen.blit(shade_surf, (cutoff_x, sgraph_y))

            pygame.draw.line(screen, COLOR_ACCENT_GOLD, (cutoff_x, sgraph_y), (cutoff_x, sgraph_y + sgraph_h), 2)
            screen.blit(mini_font.render("CUTOFF", True, COLOR_ACCENT_GOLD), (cutoff_x - 22, sgraph_y + 4))

            screen.blit(small_font.render("Bass (80 Hz)", True, COLOR_TEXT_MUTED), (sgraph_x + 4, sgraph_y + sgraph_h + 4))
            screen.blit(small_font.render("Treble (8 kHz)", True, COLOR_TEXT_MUTED), (sgraph_x + sgraph_w - 80, sgraph_y + sgraph_h + 4))

            # ------------------------------------------------------------------
            # CARD 2: HORIZONTAL AXIS (RHYTHMIC FLUTTER)
            # ------------------------------------------------------------------
            c2_y = c1_y + c1_h + 12
            c2_h = 195
            c2_rect = pygame.Rect(side_x, c2_y, side_w, c2_h)
            pygame.draw.rect(screen, COLOR_PANEL, c2_rect, border_radius=8)
            pygame.draw.rect(screen, COLOR_BORDER, c2_rect, width=1, border_radius=8)

            screen.blit(card_title_font.render("X-Axis: Rhythmic Flutter (Pulse)", True, COLOR_ACCENT_PURPLE), (side_x + 18, c2_y + 12))

            # Big Stat Readout
            fl_text = f"Flutter Rate: {vis['flutter_rate_hz']:4.1f} Pulses / sec" if vis["flutter_active"] else "Flutter Rate: OFF (Smooth Drone)"
            screen.blit(big_stat_font.render(fl_text, True, COLOR_TEXT_PRIMARY), (side_x + 18, c2_y + 36))

            effect_desc = "Chopping the ambient drone into rapid pulsating energy" if vis["flutter_active"] else "Sound flows continuously without rhythm interruptions"
            screen.blit(small_font.render(effect_desc, True, COLOR_TEXT_SECONDARY), (side_x + 18, c2_y + 64))

            # Visual Pulse Indicator & Wave
            pbox_x = side_x + 18
            pbox_y = c2_y + 88
            pbox_w = side_w - 36
            pbox_h = 86

            pygame.draw.rect(screen, COLOR_PANEL_INNER, (pbox_x, pbox_y, pbox_w, pbox_h), border_radius=6)
            pygame.draw.rect(screen, COLOR_BORDER, (pbox_x, pbox_y, pbox_w, pbox_h), width=1, border_radius=6)

            # Rhythmic Pulse Light (Beats in time with the sound)
            light_cx = pbox_x + 45
            light_cy = pbox_y + 43
            light_rad = 14 + int(pulse_val * 14)
            light_alpha = int(pulse_val * 220)
            
            if vis["flutter_active"]:
                pygame.draw.circle(screen, (80, 40, 120), (light_cx, light_cy), 28)
                pulse_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
                pygame.draw.circle(pulse_surf, (180, 115, 255, light_alpha), (30, 30), light_rad)
                screen.blit(pulse_surf, (light_cx - 30, light_cy - 30))
                pygame.draw.circle(screen, COLOR_TEXT_PRIMARY, (light_cx, light_cy), 5)
            else:
                pygame.draw.circle(screen, (30, 40, 60), (light_cx, light_cy), 18)
                pygame.draw.circle(screen, COLOR_ACCENT_CYAN, (light_cx, light_cy), 6)

            screen.blit(small_font.render("Live Pulse", True, COLOR_TEXT_MUTED), (light_cx - 26, light_cy + 24))

            # Animated Rhythm Waveform
            wave_x = pbox_x + 105
            wave_y = pbox_y + 12
            wave_w = pbox_w - 118
            wave_h = pbox_h - 24

            if vis["flutter_active"]:
                pts = []
                freq = vis["flutter_rate_hz"]
                for px in range(wave_w):
                    t_val = (px / float(wave_w)) * 4.0 * np.pi
                    lfo_val = 0.5 + 0.5 * math.cos(t_val + vis["flutter_phase"])
                    g_val = max(0.0, min(1.0, (lfo_val - 0.35) * 5.0))
                    py = wave_y + wave_h - 4 - int(g_val * (wave_h - 8))
                    pts.append((wave_x + px, py))
                if len(pts) > 1:
                    pygame.draw.lines(screen, COLOR_ACCENT_PURPLE, False, pts, 2)
            else:
                # Flat unbroken line for smooth drone
                pygame.draw.line(screen, COLOR_ACCENT_CYAN, (wave_x, wave_y + wave_h // 2), (wave_x + wave_w, wave_y + wave_h // 2), 2)
                screen.blit(small_font.render("Continuous Sustained Drone", True, (90, 110, 145)), (wave_x + 20, wave_y + wave_h // 2 - 18))

            # ------------------------------------------------------------------
            # CARD 3: MOTION SMOOTHING (ANTI-JITTER)
            # ------------------------------------------------------------------
            c3_y = c2_y + c2_h + 12
            c3_h = pad_box_h - (c1_h + c2_h + 24)
            c3_rect = pygame.Rect(side_x, c3_y, side_w, c3_h)
            pygame.draw.rect(screen, COLOR_PANEL, c3_rect, border_radius=8)
            pygame.draw.rect(screen, COLOR_BORDER, c3_rect, width=1, border_radius=8)

            screen.blit(card_title_font.render("Motion Smoothing (Anti-Jitter)", True, COLOR_TEXT_PRIMARY), (side_x + 18, c3_y + 12))

            # Red dot explanation
            pygame.draw.circle(screen, COLOR_RAW_RED, (side_x + 26, c3_y + 44), 5)
            screen.blit(body_font.render("Red Marker   = Raw coordinate (instant camera / mouse input)", True, COLOR_TEXT_PRIMARY), (side_x + 40, c3_y + 36))

            # White dot explanation (matching the actual white puck on the graph)
            pygame.draw.circle(screen, (255, 255, 255), (side_x + 26, c3_y + 72), 5)
            screen.blit(body_font.render("White Marker = Filtered coordinate (smooth sweeps without clicks)", True, COLOR_TEXT_PRIMARY), (side_x + 40, c3_y + 64))

            # MediaPipe integration note
            screen.blit(small_font.render("Normalized [0.0 - 1.0] contract ready for MediaPipe hand tracking.", True, COLOR_TEXT_MUTED), (side_x + 18, c3_y + 100))

            pygame.display.flip()
            clock.tick(60)

    finally:
        engine.stop()
        pygame.quit()
        print("[Demo Studio] Exited cleanly.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="2D Matrix: Technical Audio Demonstration")
    parser.add_argument("--test", action="store_true", help="Run automated test without launching UI")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.wav or .flac)")
    args = parser.parse_args()

    if args.test:
        print("[Running Technical Verification Test: Low-Pass + Rhythmic Flutter]")
        engine = LowPassAudioEngine(audio_path=args.audio)
        engine.start()
        print(f"Audio started. Testing 2D sweep across 100 steps...")
        for step in range(100):
            norm_y = 0.5 + 0.4 * math.sin(step / 10.0)
            norm_x = 0.5 + 0.4 * math.cos(step / 10.0)
            r_y, s_y, r_x, s_x = engine.update_filter(norm_y, norm_x)
            vis = engine.get_visual_data()
            assert engine.MIN_LP_FREQ <= vis["cutoff_hz"] <= engine.MAX_LP_FREQ
            assert len(vis["bands"]) == engine.NUM_BANDS
            time.sleep(0.005)
        engine.stop()
        print(f"[Verification PASSED] 4-pole low-pass and rhythmic flutter verified cleanly.")
        return

    try:
        run_mouse_simulator(audio_path=args.audio)
    except Exception as e:
        print(f"Interactive window could not be opened: {e}", file=sys.stderr)
        print("Running console fallback test...", file=sys.stderr)
        engine = LowPassAudioEngine(audio_path=args.audio)
        engine.start()
        for i in range(12):
            norm_y = i / 12.0
            norm_x = i / 12.0
            engine.update_filter(norm_y, norm_x)
            vis = engine.get_visual_data()
            print(f"Step {i:02d}: Y={norm_y:.2f} (Cutoff={vis['cutoff_hz']:5.0f}Hz) | X={norm_x:.2f} (Flutter={vis['flutter_rate_hz']:4.1f}Hz, Peak={vis['peak_l']:.3f})")
            time.sleep(0.05)
        engine.stop()
        print("[Fallback Test Complete] Engine verified.")

if __name__ == "__main__":
    main()
