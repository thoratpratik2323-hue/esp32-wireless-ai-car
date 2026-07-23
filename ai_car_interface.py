import pygame
import sys
import os
import csv
import time
import math
import socket
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Initialize Pygame
pygame.init()
pygame.font.init()

# --- WINDOW CONFIGURATION ---
# Open a resizable window first
screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
pygame.display.set_caption("ESP32 Wireless Car Control Panel & Simulator")
clock = pygame.time.Clock()

# Maximize the window via SDL2 Window module
try:
    from pygame._sdl2 import Window
    Window.from_display_module().maximize()
except Exception:
    pass

# Allow the OS window manager a split second to maximize and report new sizes
pygame.event.pump()
time.sleep(0.1)
pygame.event.pump()

WIDTH, HEIGHT = screen.get_size()

# --- DYNAMIC SAFE LAYOUT (Prevents clipping on taskbar/edges) ---
SAFE_HEIGHT = HEIGHT - 45   # Leave 45px safety margin at the bottom (for taskbar)
sim_w = WIDTH - 340         # Drive simulation area width
dash_x = sim_w + 15         # Start dashboard panel with a 15px gap
dash_w = 300                # Set dashboard width (leaves ~25px safety margin on the far right)

# --- CLEAN FLAT COLORS ---
BG_COLOR = (240, 242, 245)      # Light gray background
PANEL_BG = (255, 255, 255)      # Pure white panels
BORDER_COLOR = (218, 220, 224)  # Soft gray border
TEXT_COLOR = (32, 33, 36)       # Dark charcoal text
MUTED_TEXT = (95, 99, 104)      # Medium gray text
COLOR_BLUE = (26, 115, 232)     # Google Blue
COLOR_GREEN = (30, 142, 62)     # Google Green
COLOR_RED = (217, 48, 37)       # Google Red
CAR_COLOR = (26, 115, 232)      # Blue car

# --- FONTS ---
font_title = pygame.font.SysFont("Arial", 22, bold=True)
font_header = pygame.font.SysFont("Arial", 16, bold=True)
font_body = pygame.font.SysFont("Arial", 14)
font_stats = pygame.font.SysFont("Arial", 24, bold=True)

# --- SYSTEM STATES ---
STATE_MANUAL = "Manual Control (Logging Data)"
STATE_AUTO = "Autonomous Control (AI)"
current_state = STATE_MANUAL

# Logging and ML configuration
DATA_FILE = "training_data.csv"
model = None
model_trained = False

# --- WI-FI TCP SOCKET SERVER SETUP ---
def get_laptop_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

LAPTOP_IP = get_laptop_ip()
PORT = 5005

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)
    server_socket.setblocking(False)
    socket_initialized = True
except Exception as e:
    socket_initialized = False
    print(f"Failed to bind socket: {e}")

client_socket = None
client_address = None
esp32_connected = False

# --- SYSTEM LOGGING SYSTEM ---
system_logs = []
log_scroll_offset = 0
prev_status_message = ""

def add_log(msg, color=TEXT_COLOR):
    global log_scroll_offset
    timestamp = time.strftime("%H:%M:%S")
    system_logs.append((f"[{timestamp}] {msg}", color))
    if len(system_logs) > 200:
        system_logs.pop(0)
    # Auto-scroll to bottom on new log
    log_panel_h = SAFE_HEIGHT - 575
    visible = max(1, (log_panel_h - 45) // 18)
    log_scroll_offset = max(0, len(system_logs) - visible)

# --- MOCK SIMULATOR SETUP ---
# Simple rectangular obstacles representing walls and blocks
obstacles = [
    # Boundary Walls (constraining movement within SAFE_HEIGHT)
    pygame.Rect(0, 0, sim_w, 10),
    pygame.Rect(0, SAFE_HEIGHT - 10, sim_w, 10),
    pygame.Rect(0, 0, 10, SAFE_HEIGHT),
    pygame.Rect(sim_w - 10, 0, 10, SAFE_HEIGHT),
    
    # Internal Obstacles
    pygame.Rect(150, 120, 100, 140),
    pygame.Rect(380, 100, 80, 180),
    pygame.Rect(200, 380, 200, 60),
    pygame.Rect(80, 440, 60, 80),
    pygame.Rect(460, 320, 60, 160),
]

class VirtualCar:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = -90
        self.speed = 0
        self.max_speed = 3.0
        self.steer_speed = 3.0
        self.width = 24
        self.height = 16
        
        # Sensor angles
        self.sensor_angles = [-40, 0, 40]
        self.distances = [250.0, 250.0, 250.0]

    def update(self, keys, ai_command=None):
        drive = 0
        steer = 0
        
        if ai_command is not None:
            if ai_command == "F":
                drive = 1
            elif ai_command == "L":
                drive = 0.4
                steer = -1
            elif ai_command == "R":
                drive = 0.4
                steer = 1
            elif ai_command == "B":
                drive = -1
            elif ai_command == "S":
                drive = 0
        else:
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                drive = 1
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                drive = -1
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                steer = -1
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                steer = 1

        self.speed = drive * self.max_speed
        self.angle += steer * self.steer_speed
        
        rad = math.radians(self.angle)
        new_x = self.x + self.speed * math.cos(rad)
        new_y = self.y + self.speed * math.sin(rad)
        
        # Collision check
        temp_rect = pygame.Rect(new_x - self.width/2, new_y - self.height/2, self.width, self.height)
        collision = False
        for obs in obstacles:
            if temp_rect.colliderect(obs):
                collision = True
                break
                
        if not collision:
            self.x = new_x
            self.y = new_y
        else:
            self.speed = 0
            
        self.cast_rays()

    def cast_rays(self):
        self.distances = []
        for s_angle in self.sensor_angles:
            ray_angle = math.radians(self.angle + s_angle)
            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)
            
            start_x = self.x
            start_y = self.y
            dist = 250.0
            
            for d in np.arange(0, 250, 4.0):
                px = start_x + d * dx
                py = start_y + d * dy
                
                hit = False
                for obs in obstacles:
                    if obs.collidepoint(px, py):
                        dist = d
                        hit = True
                        break
                if hit:
                    break
            self.distances.append(round(dist))

    def draw(self, surface):
        for i, s_angle in enumerate(self.sensor_angles):
            ray_angle = math.radians(self.angle + s_angle)
            end_x = self.x + self.distances[i] * math.cos(ray_angle)
            end_y = self.y + self.distances[i] * math.sin(ray_angle)
            
            color = COLOR_RED if self.distances[i] < 60 else COLOR_GREEN
            pygame.draw.line(surface, color, (self.x, self.y), (end_x, end_y), 1)
            pygame.draw.circle(surface, color, (int(end_x), int(end_y)), 3)

        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, CAR_COLOR, (0, 0, self.width, self.height), border_radius=2)
        pygame.draw.line(car_surface, (255, 255, 255), (self.width - 4, 0), (self.width - 4, self.height), 2)
        
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)
        new_rect = rotated_car.get_rect(center=(self.x, self.y))
        surface.blit(rotated_car, new_rect.topleft)

virtual_car = VirtualCar(80, 80)
last_action = "S"
recorded_count = 0

# --- DATA CSV LOGGING ---
def check_csv():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["dist_left", "dist_center", "dist_right", "action"])

check_csv()

def get_row_count():
    if not os.path.exists(DATA_FILE):
        return 0
    try:
        df = pd.read_csv(DATA_FILE)
        return len(df)
    except:
        return 0

recorded_count = get_row_count()

def record_data(d_left, d_center, d_right, action):
    global recorded_count
    if action == "S":
        return
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([d_left, d_center, d_right, action])
    recorded_count += 1

# --- MACHINE LEARNING MODEL ---
def train_model():
    global model, model_trained
    if not os.path.exists(DATA_FILE) or get_row_count() < 10:
        return False, "Error: Needs at least 10 logged dataset entries."
    try:
        df = pd.read_csv(DATA_FILE)
        X = df[["dist_left", "dist_center", "dist_right"]].values
        y = df["action"].values
        model = DecisionTreeClassifier(max_depth=4)
        model.fit(X, y)
        with open("car_ai_model.pkl", "wb") as f:
            pickle.dump(model, f)
        model_trained = True
        return True, "Success: Decision Tree Model trained!"
    except Exception as e:
        return False, f"Error: {e}"

if os.path.exists("car_ai_model.pkl"):
    try:
        with open("car_ai_model.pkl", "rb") as f:
            model = pickle.load(f)
        model_trained = True
    except:
        pass

# --- SIMPLE CONTAINER DRAWING ---
def draw_simple_panel(surface, title, rect):
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=4)
    pygame.draw.rect(surface, BORDER_COLOR, rect, width=1, border_radius=4)
    title_lbl = font_header.render(title, True, TEXT_COLOR)
    surface.blit(title_lbl, (rect.x + 12, rect.y + 10))

# --- MAIN LOOP ---
status_message = "Wi-Fi Server active. Waiting for ESP32..."
message_color = COLOR_BLUE

while True:
    screen.fill(BG_COLOR)
    keys = pygame.key.get_pressed()
    
    # Observer: Append status messages to scrollable logs when they change
    if status_message != prev_status_message:
        add_log(status_message, message_color)
        prev_status_message = status_message
    
    # Check for incoming ESP32 connection (non-blocking)
    if socket_initialized and not esp32_connected:
        try:
            client_socket, client_address = server_socket.accept()
            client_socket.setblocking(False)
            esp32_connected = True
            status_message = f"Connected to ESP32: {client_address[0]}"
            message_color = COLOR_GREEN
        except BlockingIOError:
            pass

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if client_socket:
                client_socket.close()
            server_socket.close()
            pygame.quit()
            sys.exit()
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if client_socket:
                    client_socket.close()
                server_socket.close()
                pygame.quit()
                sys.exit()
                
            if event.key == pygame.K_m:
                current_state = STATE_MANUAL
                status_message = "Mode: Manual Logging active."
                message_color = COLOR_BLUE
            elif event.key == pygame.K_o:
                if model_trained:
                    current_state = STATE_AUTO
                    status_message = "Mode: Autonomous Self-Driving active."
                    message_color = COLOR_GREEN
                else:
                    status_message = "Error: Please train the model first (Press T)."
                    message_color = COLOR_RED
            
            if event.key == pygame.K_t:
                success, msg = train_model()
                status_message = msg
                message_color = COLOR_GREEN if success else COLOR_RED
                
            if event.key == pygame.K_c:
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                check_csv()
                recorded_count = 0
                status_message = "Logged dataset has been cleared."
                message_color = COLOR_RED

        elif event.type == pygame.MOUSEBUTTONDOWN:
            log_rect = pygame.Rect(dash_x, 575, dash_w, SAFE_HEIGHT - 575)
            if log_rect.collidepoint(event.pos):
                if event.button == 4: # Scroll Up (Mouse Wheel Up)
                    log_scroll_offset = max(0, log_scroll_offset - 1)
                elif event.button == 5: # Scroll Down (Mouse Wheel Down)
                    log_panel_h = SAFE_HEIGHT - 575
                    visible = max(1, (log_panel_h - 45) // 18)
                    max_scroll = max(0, len(system_logs) - visible)
                    log_scroll_offset = min(max_scroll, log_scroll_offset + 1)

    # --- SENSOR & DRIVE LOGIC ---
    current_distances = [250.0, 250.0, 250.0]
    
    if esp32_connected:
        # Read from Wireless ESP32 Socket
        try:
            data = client_socket.recv(1024).decode('utf-8').strip()
            if not data:
                # Connection closed
                client_socket.close()
                client_socket = None
                esp32_connected = False
                status_message = "ESP32 client connection closed."
                message_color = COLOR_RED
            else:
                # Read latest package from buffer
                lines = data.split('\n')
                latest_line = lines[-1].strip()
                parts = latest_line.split(',')
                if len(parts) == 3:
                    current_distances = [float(p) for p in parts]
        except BlockingIOError:
            pass
        except Exception as e:
            client_socket.close()
            client_socket = None
            esp32_connected = False
            status_message = f"Connection error: {e}"
            message_color = COLOR_RED
            
        if current_state == STATE_MANUAL:
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                last_action = "F"
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                last_action = "L"
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                last_action = "R"
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                last_action = "B"
            else:
                last_action = "S"
            
            # Write steering command back to client
            if client_socket:
                try:
                    client_socket.send(f"{last_action}\n".encode())
                except:
                    pass
                
            if last_action != "S":
                record_data(current_distances[0], current_distances[1], current_distances[2], last_action)
                
        elif current_state == STATE_AUTO:
            if model_trained:
                pred = model.predict([current_distances])[0]
                last_action = pred
                if client_socket:
                    try:
                        client_socket.send(f"{last_action}\n".encode())
                    except:
                        pass
            else:
                current_state = STATE_MANUAL
    else:
        # Mock mode simulator update
        if current_state == STATE_MANUAL:
            virtual_car.update(keys)
            current_distances = virtual_car.distances
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                last_action = "F"
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                last_action = "L"
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                last_action = "R"
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                last_action = "B"
            else:
                last_action = "S"
                
            if last_action != "S":
                record_data(current_distances[0], current_distances[1], current_distances[2], last_action)
        elif current_state == STATE_AUTO:
            if model_trained:
                pred = model.predict([virtual_car.distances])[0]
                last_action = pred
                virtual_car.update(keys, ai_command=pred)
                current_distances = virtual_car.distances
            else:
                current_state = STATE_MANUAL

    # --- RENDERING DECK ---
    
    # 1. Simple Flat Simulation Area (Dynamic Width & SAFE_HEIGHT)
    pygame.draw.rect(screen, (248, 249, 250), (0, 0, sim_w, SAFE_HEIGHT))
    
    # Draw simple flat obstacles
    for obs in obstacles:
        pygame.draw.rect(screen, (186, 191, 198), obs)
        pygame.draw.rect(screen, (138, 143, 149), obs, width=1)
        
    # Draw virtual car
    if not esp32_connected:
        virtual_car.draw(screen)
    else:
        box = pygame.Rect(sim_w // 2 - 150, SAFE_HEIGHT // 2 - 50, 300, 100)
        pygame.draw.rect(screen, PANEL_BG, box, border_radius=4)
        pygame.draw.rect(screen, BORDER_COLOR, box, width=1, border_radius=4)
        lbl = font_header.render("PHYSICAL WIRELESS CAR ACTIVE", True, COLOR_GREEN)
        lbl_rect = lbl.get_rect(center=(sim_w // 2, SAFE_HEIGHT // 2 - 20))
        screen.blit(lbl, lbl_rect)
        
        lbl_sub = font_body.render("Steering commands routing over Wi-Fi...", True, MUTED_TEXT)
        lbl_sub_rect = lbl_sub.get_rect(center=(sim_w // 2, SAFE_HEIGHT // 2 + 10))
        screen.blit(lbl_sub, lbl_sub_rect)

    # Separation Line
    pygame.draw.line(screen, BORDER_COLOR, (sim_w, 0), (sim_w, SAFE_HEIGHT), 2)
    
    # 2. Right Side Simple Dashboard Panel (Dynamic Position)
    screen.blit(font_title.render("Telemetry Control Panel", True, TEXT_COLOR), (dash_x, 15))
    screen.blit(font_body.render("ESP32 Wi-Fi Socket ML Setup", True, MUTED_TEXT), (dash_x, 40))
    pygame.draw.line(screen, BORDER_COLOR, (dash_x, 60), (dash_x + dash_w - 5, 60), 1)
    
    # Panel 1: Connection Info
    draw_simple_panel(screen, "Wi-Fi Server Config", pygame.Rect(dash_x, 75, dash_w, 95))
    if esp32_connected:
        screen.blit(font_body.render(f"Status: Connected", True, COLOR_GREEN), (dash_x + 15, 110))
        screen.blit(font_body.render(f"ESP32 IP: {client_address[0]}", True, TEXT_COLOR), (dash_x + 15, 130))
    else:
        screen.blit(font_body.render("Status: Offline (Mock Simulation)", True, MUTED_TEXT), (dash_x + 15, 110))
        screen.blit(font_body.render(f"Server IP: {LAPTOP_IP} : {PORT}", True, TEXT_COLOR), (dash_x + 15, 130))

    # Panel 2: Sensor Telemetry
    draw_simple_panel(screen, "Sensor Readings (Distance)", pygame.Rect(dash_x, 185, dash_w, 115))
    screen.blit(font_body.render(f"Left Sensor:      {int(current_distances[0])} cm", True, TEXT_COLOR), (dash_x + 15, 220))
    screen.blit(font_body.render(f"Center Sensor:  {int(current_distances[1])} cm", True, TEXT_COLOR), (dash_x + 15, 240))
    screen.blit(font_body.render(f"Right Sensor:    {int(current_distances[2])} cm", True, TEXT_COLOR), (dash_x + 15, 260))

    # Panel 3: Machine Learning Status
    draw_simple_panel(screen, "Machine Learning Engine", pygame.Rect(dash_x, 315, dash_w, 130))
    screen.blit(font_stats.render(f"{recorded_count}", True, COLOR_BLUE), (dash_x + 15, 345))
    screen.blit(font_body.render("Logged Data Samples", True, MUTED_TEXT), (dash_x + 80, 350))
    
    model_status_t = "Trained & Ready" if model_trained else "Untrained"
    model_status_c = COLOR_GREEN if model_trained else COLOR_RED
    screen.blit(font_body.render(f"AI Model Status: {model_status_t}", True, model_status_c), (dash_x + 15, 380))
    
    state_lbl = font_body.render(f"Active Mode: {current_state}", True, TEXT_COLOR)
    screen.blit(state_lbl, (dash_x + 15, 400))
    
    cmd_lbl = font_body.render(f"Last Command: {last_action}", True, MUTED_TEXT)
    screen.blit(cmd_lbl, (dash_x + 15, 420))
    
    # Panel 4: Keyboard Controls
    draw_simple_panel(screen, "Keyboard Controls", pygame.Rect(dash_x, 455, dash_w, 110))
    screen.blit(font_body.render("[WASD] / Arrows : Drive Manual Car", True, TEXT_COLOR), (dash_x + 15, 485))
    screen.blit(font_body.render("[M] Manual Mode  |  [O] Auto AI Mode", True, TEXT_COLOR), (dash_x + 15, 505))
    screen.blit(font_body.render("[T] Train Model     |  [C] Clear Logs", True, TEXT_COLOR), (dash_x + 15, 525))
    screen.blit(font_body.render("[ESC] Close App     |  Active Server port: 5005", True, MUTED_TEXT), (dash_x + 15, 545))
    
    # Bottom Status Bar Panel (System Log)
    log_panel_y = 575
    log_panel_height = SAFE_HEIGHT - log_panel_y
    status_rect = pygame.Rect(dash_x, log_panel_y, dash_w, log_panel_height)
    pygame.draw.rect(screen, PANEL_BG, status_rect, border_radius=4)
    pygame.draw.rect(screen, BORDER_COLOR, status_rect, width=1, border_radius=4)
    
    screen.blit(font_header.render("SYSTEM LOG:", True, TEXT_COLOR), (dash_x + 12, log_panel_y + 10))
    
    # Calculate visible lines inside the log box area (leaving 45px for header/border)
    inner_log_h = log_panel_height - 45
    max_visible_lines = max(1, inner_log_h // 18)
    
    # Set clipping region to prevent text drawing outside the panel boundaries
    clip_rect = pygame.Rect(dash_x + 12, log_panel_y + 35, dash_w - 24, inner_log_h)
    screen.set_clip(clip_rect)
    
    # Draw logs
    start_idx = log_scroll_offset
    end_idx = min(len(system_logs), start_idx + max_visible_lines)
    
    for i in range(start_idx, end_idx):
        log_text, log_color = system_logs[i]
        line_y = log_panel_y + 35 + (i - start_idx) * 18
        lbl = font_body.render(log_text, True, log_color)
        screen.blit(lbl, (dash_x + 12, line_y))
        
    # Reset clip
    screen.set_clip(None)
    
    # Draw scrollbar if needed
    if len(system_logs) > max_visible_lines:
        scrollbar_x = dash_x + dash_w - 10
        scrollbar_y = log_panel_y + 35
        scrollbar_h = inner_log_h
        
        pygame.draw.line(screen, BORDER_COLOR, (scrollbar_x, scrollbar_y), (scrollbar_x, scrollbar_y + scrollbar_h), 1)
        
        fraction = max_visible_lines / len(system_logs)
        thumb_h = max(15, int(scrollbar_h * fraction))
        
        scroll_fraction = log_scroll_offset / (len(system_logs) - max_visible_lines)
        thumb_y = scrollbar_y + int((scrollbar_h - thumb_h) * scroll_fraction)
        
        pygame.draw.rect(screen, MUTED_TEXT, (scrollbar_x - 3, thumb_y, 6, thumb_h), border_radius=2)

    pygame.display.flip()
    clock.tick(60)
