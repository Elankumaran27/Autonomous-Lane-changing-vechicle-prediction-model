# Required libraries
import pygame
import random
import pickle
import lightgbm as lgb
import numpy as np
import joblib

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
LANE_WIDTH = WIDTH // 3
FPS = 30
CAR_LENGTH = 40
CAR_WIDTH = 20
SAFE_DISTANCE = 50  # Pixels
BRAKE_DECELERATION = 0.05  # Speed reduction per frame if braking
ACCELERATION = 0.02         # Speed increase if road is clear
MAX_SPEED = 5               # Max speed limit

# Colors
WHITE = (255, 255, 255)
GREY = (50, 50, 50)
RED = (255, 0, 0)
BLUE = (0, 150, 255)

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Lane Change Simulation")
clock = pygame.time.Clock()

# Load models
with open('gmm_model.pkl', 'rb') as f:
    gmm_model = pickle.load(f)

scaler = joblib.load('scaler.pkl')
lgb_model = lgb.Booster(model_file='final_lane_change_model.txt')


# Car Class
class Car:
    def __init__(self, vehicle_id, lane, y, speed):
        self.vehicle_id = vehicle_id
        self.lane = lane
        self.y = y
        self.x = self.lane * LANE_WIDTH + LANE_WIDTH // 2
        self.speed = speed
        self.color = BLUE
        self.target_lane = None
        self.lane_change_in_progress = False
        self.braking = False

    def move(self):
        # 🚗 Apply braking or acceleration
        if self.braking:
            self.speed = max(self.speed - BRAKE_DECELERATION, 0)
        else:
            self.speed = min(self.speed + ACCELERATION, MAX_SPEED)

        self.y -= self.speed

        if self.lane_change_in_progress:
            self.perform_lane_change()

    def perform_lane_change(self):
        target_x = self.target_lane * LANE_WIDTH + LANE_WIDTH // 2
        if abs(self.x - target_x) > 5:
            if self.x < target_x:
                self.x += 5
            else:
                self.x -= 5
        else:
            self.x = target_x
            self.lane = self.target_lane
            self.lane_change_in_progress = False
            self.target_lane = None
            self.color = BLUE

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x - CAR_WIDTH//2, self.y - CAR_LENGTH//2, CAR_WIDTH, CAR_LENGTH))

    def predict_lane_change(self, surrounding_info):
        feature_vector = np.array(surrounding_info).reshape(1, -1)
        feature_vector = scaler.transform(feature_vector)
        prediction = lgb_model.predict(feature_vector)
        return prediction[0] > 0.5


# Traffic Manager
class TrafficManager:
    def __init__(self):
        self.cars = []
        self.spawn_interval = 700  # ms
        self.last_spawn_time = pygame.time.get_ticks()
        self.vehicle_count = 0

    def spawn_car(self):
        lane = random.choice([0, 1, 2])
        safe_spawn = False
        attempts = 0
        while not safe_spawn and attempts < 10:
            y = HEIGHT + random.randint(0, 100)
            safe_spawn = True
            for other_car in self.cars:
                if other_car.lane == lane and abs(other_car.y - y) < SAFE_DISTANCE:
                    safe_spawn = False
                    break
            attempts += 1

        if safe_spawn:
            speed = random.randint(2, 4)
            car = Car(self.vehicle_count, lane, y, speed)
            self.cars.append(car)
            self.vehicle_count += 1

    def update(self):
        # 🚗 Spawn new cars periodically
        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time > self.spawn_interval:
            self.spawn_car()
            self.last_spawn_time = current_time

        for car in self.cars:
            car.move()

        self.handle_lane_changes()

        # Clean up off-screen cars
        self.cars = [car for car in self.cars if car.y + CAR_LENGTH > 0]

    def handle_lane_changes(self):
        for car in self.cars:
            if car.lane_change_in_progress:
                continue  # Skip already changing

            front_car = self.get_front_car(car)
            car.braking = False  # Reset each update

            if front_car:
                gap = car.y - front_car.y
                if gap < SAFE_DISTANCE * 1.5 and car.speed > front_car.speed:
                    # 🚧 Try to change lane if blocked
                    lane_changed = False
                    for direction in [-1, 1]:  # Left then right
                        target_lane = car.lane + direction
                        if 0 <= target_lane <= 2 and self.is_lane_safe(car, target_lane):
                            features = self.create_feature_vector(car, front_car)
                            if car.predict_lane_change(features):
                                car.target_lane = target_lane
                                car.lane_change_in_progress = True
                                car.color = RED
                                lane_changed = True
                                break

                    if not lane_changed:
                        # 🛑 Apply brake if no escape
                        car.braking = True
                else:
                    # 🚗 Accelerate if no need to brake
                    car.braking = False
            else:
                # 🛣️ Road clear ahead
                car.braking = False

    def get_front_car(self, car):
        # 🕵️ Get closest car ahead in same lane
        candidates = [c for c in self.cars if c.lane == car.lane and c.y < car.y]
        candidates.sort(key=lambda c: car.y - c.y)
        return candidates[0] if candidates else None

    def is_lane_safe(self, car, target_lane, prediction_time=70):
        # 🔮 Predict if other cars will be in collision path after lane change
        car_future_y = car.y
        car_speed = car.speed

        for other_car in self.cars:
            if other_car.lane == target_lane:
                other_future_y = other_car.y
                other_speed = other_car.speed

                for _ in range(prediction_time):
                    car_future_y -= car_speed
                    other_future_y -= other_speed
                    if other_future_y > car_future_y and (other_future_y - car_future_y) < SAFE_DISTANCE + 20:
                        return False  # Another car behind could crash into us

                    if abs(car_future_y - other_future_y) < SAFE_DISTANCE +10:
                        return False
        return True

    def create_feature_vector(self, car, front_car):
        gap = car.y - front_car.y
        rel_vel = car.speed - front_car.speed
        rel_acc = 0  # Simplified
        features = [
            car.vehicle_id, 0, car.lane + 1, car.speed, 0,
            car.x, car.y, car.x, car.y,
            gap, front_car.vehicle_id,
            1 if car.speed != 0 else 999,
            rel_acc, rel_vel, 2, 0, gap, 1
        ]
        return features


# Main Loop
def main():
    running = True
    traffic = TrafficManager()

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        traffic.update()

        screen.fill(WHITE)
        draw_lanes()
        for car in traffic.cars:
            car.draw(screen)

        pygame.display.flip()

    pygame.quit()

def draw_lanes():
    for i in range(1, 3):
        pygame.draw.line(screen, GREY, (i * LANE_WIDTH, 0), (i * LANE_WIDTH, HEIGHT), 5)

# Start simulation
if __name__ == "__main__":
    main()
