# AI/ML Smart Car - Python Dashboard & Simulator

This is the Python interface for your AI/ML Smart Car. It serves two purposes:
1. **Mock Simulator Mode:** Test the machine learning pipeline on your laptop immediately without hardware.
2. **Physical Control Mode:** Connect to your physical Arduino Uno via Serial/USB, collect sensor data, train models, and steer the physical car using AI.

---

## 🚀 Setup & Running

### 1. Install Dependencies
You need Python 3 installed on your computer. Open your terminal/command prompt, navigate to this project folder, and run:

```bash
pip install -r requirements.txt
```

### 2. Run the Interface
To start the dashboard and simulator, run:

```bash
python ai_car_interface.py
```

---

## 🎮 How to Use (Keyboard Controls)

### Control Modes
* **Press `M`:** Switch to **Manual Mode** (Data Logging is active).
* **Press `O`:** Switch to **Autonomous Mode** (The AI Model takes control).
* **Press `T`:** **Train the AI Model** on the logged data in real-time.
* **Press `C`:** **Clear** all logged training data.

### Driving (Manual Mode)
Use the **W, A, S, D** or **Arrow Keys** to steer the car:
* `W` / `Up Arrow`: Move Forward
* `A` / `Left Arrow`: Turn Left
* `D` / `Right Arrow`: Turn Right
* `S` / `Down Arrow`: Move Backward / Reverse
* Release all keys to Stop

---

## 🤖 How the Machine Learning Works

1. **Collect Training Data:**
   * Run the script in **Mock Mode** (Manual Mode).
   * Drive the car around the screen avoiding the gray boxes. 
   * As you drive, the program logs your three sensor readings (Left distance, Center distance, Right distance) and your key press (Forward, Left, Right) to `training_data.csv`.
   * Try to log at least **300–500 samples** of good driving.

2. **Train the AI Model:**
   * Press `T` on your keyboard.
   * The script loads the CSV file and trains a **Decision Tree Classifier** from `scikit-learn` in less than a second!

3. **Enable Self-Driving:**
   * Press `O`.
   * The car is now controlled by the Decision Tree model! It reads its distance sensors and chooses the steering direction based on what it learned from your driving.
