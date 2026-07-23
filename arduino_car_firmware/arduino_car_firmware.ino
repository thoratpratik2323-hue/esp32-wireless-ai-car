/*
 * ESP32 Wireless AI/ML Smart Car Firmware (Web Interface + Python ML)
 * 
 * This firmware operates in two modes:
 * 1. Web Dashboard Mode: Open http://<ESP32_IP>/ in any browser (phone or laptop) 
 *    to steer the car manually via touch buttons and see real-time distance readings.
 * 2. Python ML Mode: Streams distance sensor values to Python (Port 5005) and 
 *    receives real-time steering commands from your trained Decision Tree AI model.
 * 
 * WARNING: The HC-SR04 Echo pin outputs 5V. Use a voltage divider to 
 * drop the signal to 3.3V before connecting to ESP32 GPIO 32.
 */

#include <WiFi.h>
#include <WebServer.h>

// --- WI-FI NETWORK CONFIGURATION ---
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// --- PYTHON TCP SERVER CONFIGURATION ---
const char* serverIP = "192.168.x.x"; 
const uint16_t tcpPort = 5005;

WiFiClient client;
WebServer server(80);

// --- GPIO PIN CONFIGURATION (ESP32) ---
// L298N Motor Driver Pins
const int ENA = 12;   // PWM Speed Left Motors
const int IN1 = 13;   // Dir 1 Left Motors
const int IN2 = 14;   // Dir 2 Left Motors
const int IN3 = 27;   // Dir 1 Right Motors
const int IN4 = 26;   // Dir 2 Right Motors
const int ENB = 25;   // PWM Speed Right Motors

// HC-SR04 Sonar Pins
const int TRIG_PIN = 33; // Trigger Output
const int ECHO_PIN = 32; // Echo Input (Stepped down to 3.3V via voltage divider)

// Motor Speed Setting (0 to 255)
const int MOTOR_SPEED = 200;

// Variables
float distanceValue = 250.0;
bool pythonAIMode = false; // Toggled via the Web Page

// --- WEB DASHBOARD PAGE (HTML, CSS & JS inside Flash memory) ---
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>ESP32 AI Car Controller</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #121212;
            color: #f5f5f5;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }
        h2 { margin-bottom: 5px; color: #1a73e8; }
        p { color: #9aa0a6; margin-top: 0; }
        .card {
            background-color: #1e1e1e;
            border: 1px solid #2d2d2d;
            border-radius: 8px;
            padding: 20px;
            width: 90%;
            max-width: 400px;
            box-sizing: border-box;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            text-align: center;
        }
        .sensor-val {
            font-size: 32px;
            font-weight: bold;
            color: #00e676;
            margin: 15px 0;
        }
        .sensor-val.danger { color: #ff1744; }
        
        /* Joystick style button layout */
        .controls-grid {
            display: grid;
            grid-template-columns: 80px 80px 80px;
            grid-template-rows: 80px 80px 80px;
            gap: 15px;
            justify-content: center;
            margin: 25px 0;
        }
        .btn {
            background-color: #2c2c2c;
            color: white;
            border: 2px solid #3c3c3c;
            border-radius: 50%;
            font-size: 24px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            user-select: none;
            touch-action: manipulation;
            transition: all 0.1s ease;
        }
        .btn:active {
            background-color: #1a73e8;
            border-color: #4285f4;
            transform: scale(0.95);
        }
        .btn-stop {
            background-color: #ff1744;
            border-color: #d50000;
        }
        .btn-stop:active {
            background-color: #b20000;
        }
        .mode-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px;
            background-color: #292929;
            border-radius: 6px;
            margin-top: 15px;
        }
        /* Toggle Switch */
        .switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 26px;
        }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: #555;
            transition: .3s;
            border-radius: 34px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 18px; width: 18px;
            left: 4px; bottom: 4px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }
        input:checked + .slider { background-color: #1a73e8; }
        input:checked + .slider:before { transform: translateX(24px); }
    </style>
</head>
<body>
    <div class="card">
        <h2>ESP32 Dashboard</h2>
        <p>Wireless Manual & AI Control</p>
        
        <div class="mode-container">
            <span>Python AI Mode</span>
            <label class="switch">
                <input type="checkbox" id="aiModeToggle" onchange="toggleAI()">
                <span class="slider"></span>
            </label>
        </div>

        <div class="sensor-val" id="distanceText">-- cm</div>
        <div>Sensor Distance</div>

        <div class="controls-grid" id="joystickDeck">
            <div></div>
            <div class="btn" onclick="sendCmd('F')">▲</div>
            <div></div>
            <div class="btn" onclick="sendCmd('L')">◀</div>
            <div class="btn btn-stop" onclick="sendCmd('S')">■</div>
            <div class="btn" onclick="sendCmd('R')">▶</div>
            <div></div>
            <div class="btn" onclick="sendCmd('B')">▼</div>
            <div></div>
        </div>
    </div>

    <script>
        function sendCmd(cmd) {
            fetch('/drive?dir=' + cmd);
        }

        function toggleAI() {
            var check = document.getElementById("aiModeToggle").checked;
            var val = check ? "1" : "0";
            fetch('/toggleAI?status=' + val);
            
            // Disable manual joystick buttons visual feedback if AI is driving
            var deck = document.getElementById("joystickDeck");
            if (check) {
                deck.style.opacity = "0.3";
                deck.style.pointerEvents = "none";
            } else {
                deck.style.opacity = "1";
                deck.style.pointerEvents = "auto";
            }
        }

        // Poll sensor data every 200ms
        setInterval(function() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    var el = document.getElementById("distanceText");
                    el.innerText = Math.round(data.distance) + " cm";
                    if (data.distance < 60) {
                        el.className = "sensor-val danger";
                    } else {
                        el.className = "sensor-val";
                    }
                    // Sync toggle status if updated elsewhere
                    document.getElementById("aiModeToggle").checked = data.aiMode;
                    var deck = document.getElementById("joystickDeck");
                    if (data.aiMode) {
                        deck.style.opacity = "0.3";
                        deck.style.pointerEvents = "none";
                    } else {
                        deck.style.opacity = "1";
                        deck.style.pointerEvents = "auto";
                    }
                });
        }, 200);
    </script>
</body>
</html>
)rawliteral";

// --- WEB SERVER ENDPOINT HANDLERS ---
void handleRoot() {
  server.send(200, "text/html", index_html);
}

void handleDrive() {
  if (server.hasArg("dir")) {
    String dir = server.arg("dir");
    if (!pythonAIMode) {
      executeCommand(dir[0]);
      Serial.print("Web Command: ");
      Serial.println(dir);
    }
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "Bad Request");
  }
}

void handleToggleAI() {
  if (server.hasArg("status")) {
    String status = server.arg("status");
    pythonAIMode = (status == "1");
    stopMotors(); // Safe reset
    Serial.print("AI Mode Toggled: ");
    Serial.println(pythonAIMode ? "ACTIVE" : "INACTIVE");
    server.send(200, "text/plain", "OK");
  } else {
    server.send(400, "text/plain", "Bad Request");
  }
}

void handleData() {
  String json = "{\"distance\":" + String(distanceValue) + ",\"aiMode\":" + String(pythonAIMode ? "true" : "false") + "}";
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);

  // Set motor control pins as outputs
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Set sonar pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Stop motors initially
  stopMotors();

  // Connect to Wi-Fi Network
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWi-Fi Connected successfully!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  // Configure Web Server Routes
  server.on("/", handleRoot);
  server.on("/drive", handleDrive);
  server.on("/toggleAI", handleToggleAI);
  server.on("/data", handleData);
  server.begin();
  Serial.println("HTTP Web Server started!");
}

void loop() {
  // Handle HTTP Web Client Requests
  server.handleClient();

  // 1. Read Distance Sensor (Center Sensor)
  distanceValue = readDistance(TRIG_PIN, ECHO_PIN);
  float distLeft   = 250.0; // Placeholder
  float distRight  = 250.0; // Placeholder

  // 2. Python ML Mode Routing (only run socket client if Python AI Mode is enabled)
  if (pythonAIMode) {
    if (!client.connected()) {
      Serial.print("Connecting to Python Server: ");
      Serial.println(serverIP);
      if (client.connect(serverIP, tcpPort)) {
        Serial.println("TCP Connected!");
      } else {
        Serial.println("TCP Connection failed. Retrying...");
        stopMotors();
        delay(1000);
        return;
      }
    }

    // Send Sensor Data over Wi-Fi TCP Socket
    client.print(distLeft);
    client.print(",");
    client.print(distanceValue);
    client.print(",");
    client.println(distRight);

    // Read incoming steering commands from Python
    if (client.available() > 0) {
      char command = client.read();
      executeCommand(command);
    }
  } else {
    // If not in AI mode, ensure TCP client socket is closed
    if (client.connected()) {
      client.stop();
      Serial.println("TCP Disconnected (Manual Web Mode Active)");
    }
  }

  // 50ms delay (~20Hz update rate)
  delay(50);
}

// --- ULTRASONIC SENSOR FUNCTION ---
float readDistance(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);

  long duration = pulseIn(echo, HIGH, 30000);

  if (duration == 0) {
    return 250.0;
  }

  float distance = duration * 0.034 / 2.0;

  if (distance > 250.0) distance = 250.0;
  return distance;
}

// --- MOTOR ACTUATION CONTROL ---
void executeCommand(char cmd) {
  switch (cmd) {
    case 'F': // Drive Forward
      moveForward();
      break;
    case 'B': // Drive Backward
      moveBackward();
      break;
    case 'L': // Turn Left (Spin)
      turnLeft();
      break;
    case 'R': // Turn Right (Spin)
      turnRight();
      break;
    case 'S': // Stop
    default:
      stopMotors();
      break;
  }
}

void moveForward() {
  analogWrite(ENA, MOTOR_SPEED);
  analogWrite(ENB, MOTOR_SPEED);
  
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void moveBackward() {
  analogWrite(ENA, MOTOR_SPEED);
  analogWrite(ENB, MOTOR_SPEED);
  
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void turnLeft() {
  analogWrite(ENA, MOTOR_SPEED);
  analogWrite(ENB, MOTOR_SPEED);
  
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void turnRight() {
  analogWrite(ENA, MOTOR_SPEED);
  analogWrite(ENB, MOTOR_SPEED);
  
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
