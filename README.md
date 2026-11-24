# On-Screen Soundscape Controller (Camera-Free Prototype)

Screen-based mouse simulator decoupling camera tracking from the real-time audio engine. Movement across the on-screen plane shapes the soundscape:
- **Y-Axis (Vertical):** 4-pole Low-Pass Filter ($80\,\text{Hz}$ deep underwater rumble $\to$ $8000\,\text{Hz}$ bright ambient) with dynamic make-up gain compensation.
- **X-Axis (Horizontal):** Rhythmic Flutter pulse gate ($0.0\,\text{Hz}$ smooth drone $\to$ $15.0\,\text{Hz}$ rapid flutter).
- **Coordinate Smoothing:** Exponential moving average ($85\%$ previous $+$ $15\%$ new) prevents digital clicks and zipper noise during fast movements.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <YOUR_REPO_URL>
   cd "Sound Control Testing"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

Launch the interactive 2D demonstration studio:
```bash
python main.py
# or:
python soundscape_controller.py
```

### Controls:
- **Mouse Movement on Pad:** 
  - Move vertically to sweep the filter cutoff frequency.
  - Move horizontally to accelerate the rhythmic flutter.
- **Spacebar:** Switch to the next audio stem loop.
- **Click Audio List:** Select any audio stem directly from the left panel.
- **Escape Key:** Exit application.

---

## Project Structure

```
├── audio/                   # Audio stems (WAV, FLAC, MP3)
│   ├── DarkCinematic.wav
│   ├── DreamyDrone.wav
│   ├── Drums.wav
│   ├── Forest.flac
│   ├── Harmony.mp3
│   └── Techno.wav
├── main.py                  # Standard application entry point
├── soundscape_controller.py # Core 2D audio engine & interactive Pygame UI
├── requirements.txt         # Python package dependencies
├── .gitignore               # Files excluded from Git tracking (.venv, etc.)
└── README.md                # Project documentation
```
