# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)

import math
import random
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import neat
import pygame

# Constants
WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 10 
CAR_SIZE_Y = 10

BORDER_COLOR = (255, 255, 255, 255) # White - Color To Crash on Hit
START_COLOR = (0, 255, 0, 255) #Green - Colour of Start Line
START_CV2 = [0, 255, 0] #Same as START_COLOR. Needed to use cv2.

DIRECTION = 0 
NUM_LAPS = 2 #Number of laps required before selecting best car

FILE_TYPE = ".png"

#Global variables
current_generation = 0 # Generation counter
start = [0,0]
start_angle = 180 #Start_angle = 0 for CCW, 180 for CW

class Car:

    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load('car.png').convert() # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 

        self.center = [start[0], start[1]]
        self.position = [self.center[0] - CAR_SIZE_X / 2, self.center[1] - CAR_SIZE_Y / 2]
        self.previous = [0,0]
        self.angle = start_angle
        self.speed = 0
        self.laps = 0

        self.speed_set = False # Flag For Default Speed Later on

        self.radars = [] # List For Sensors / Radars
        self.drawing_radars = [] # Radars To Be Drawn

        self.alive = True # Boolean To Check If Car is Crashed

        self.distance = 0 # Distance Driven

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position) # Draw Sprite
        self.draw_radar(screen) #OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break
            
    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 600:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate Distance To Border And Append To Radars List
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 12
            self.speed_set = True
        
        self.previous = [self.position[0], self.position[1]]

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 5px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 5)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance
        self.distance += self.speed
        
        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 5)
        self.position[1] = min(self.position[1], WIDTH - 120)
        
        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.get_laps(game_map)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values
    
    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def get_reward(self):
        # Calculate Reward (DISTANCE AND SPEED OF THE CAR TIMES A WEIGHT TO TUNE SPEED WEIGHT 
        return (self.distance / (CAR_SIZE_X / 2))*(self.speed**1.8)

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    def get_laps(self, game_map):
        xmin = int(min(self.position[0], self.previous[0]))
        xmax = int(max(self.position[0], self.previous[0]))
        ymin = int(min(self.position[1], self.previous[1]))
        ymax = int(max(self.position[1], self.previous[1]))
        for i in range(xmin,xmax+1):
            for j in range(ymin,ymax+1):
                if (game_map.get_at((i, j)) == START_COLOR):
                    self.laps += 1
                    return self.laps
        return self.laps
            
def run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    #Variable to identify fastest car
    max_laps = 0
    best_car = None

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())
    
    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load(name).convert() # Convert Speeds Up A Lot
    game_map = pygame.transform.scale(game_map, (WIDTH, HEIGHT))

    global current_generation
    current_generation += 1

    while True:
        best_car = None
        max_laps = 0
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            
        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10 # Left
            elif choice == 1:
                car.angle -= 10 # Right
            elif (choice == 2) & (car.speed > 4):
                car.speed -= 1 # Slow Down
            else:
                car.speed += 1 # Speed Up
        
        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

                # Select best car to be the first to completes 2 laps
                if car.laps == NUM_LAPS:
                    max_laps = car.laps
                    best_car = car
        #Go to next generation if all cars dead        
        if (still_alive == 0):
            break

        #Go to next generation once first car has completed 2 laps
        elif (max_laps == NUM_LAPS):
            get_raceline(nets, cars, best_car, game_map, im)
            break

        screen.blit(game_map, (0, 0))
        
        # Draw Map And All Cars That Are Alive
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        
        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (screen.get_width()/ 2, screen.get_height()/2)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen.get_width()/ 2, screen.get_height()/2+40)
        screen.blit(text, text_rect)
    
        pygame.display.flip()
        clock.tick(60) # 60 FPS
        
def get_raceline(nets, cars, best_car, game_map, im):
    raceline_x = []
    raceline_y = []
    speed = []
        
    while (best_car.laps < NUM_LAPS+1) & (best_car.is_alive()):
        #Run one extra lap
        output = nets[cars.index(best_car)].activate(best_car.get_data())
        choice = output.index(max(output))
        if choice == 0:
            best_car.angle += 10 # Left
        elif choice == 1:
            best_car.angle -= 10 # Right
        elif (choice == 2) & (best_car.speed > 4):
            best_car.speed -= 1 # Slow Down
        else:
            best_car.speed += 1 # Speed Up
        best_car.update(game_map)
        
        #Record data
        raceline_x.append(best_car.center[0])
        raceline_y.append(best_car.center[1])
        speed.append(best_car.speed)

    #Display raceline as image
    plt.imshow(im)
    
    speed_range = max(speed)-min(speed)
    print("Speed Range:", min(speed),"-", max(speed))
    print("Start speed:", speed[0])
        
    if (speed_range == 0):
        plt.scatter(raceline_x, raceline_y, color='red', marker='s', s=1)
    else:
        color = []
        for i in range(0, len(speed)):
            red = 0
            green = 0
            if speed[i]%3 == min(speed)%3:
                red = 1
            elif speed[i]%3 == (min(speed)+1)%3:
                red = 1
                green = 1
            else:
                green = 1
            color.append((red, green, 0))
        plt.scatter(raceline_x, raceline_y, c=color, marker='s', s=1)
    
    plt.show()
    
if __name__ == "__main__":

    #User identifies map file, check file type and existence
    while 1:
        name = input("Enter name of map file here: ")
        if name.endswith(FILE_TYPE):
            try:
                im = cv2.imread(name)
                plt.imshow(im)
            except TypeError:
                print("File not found, ensure that file is in the same folder as program")
            else:
                break
        else:
            print("Ensure file is type", FILE_TYPE, "and name ends with", FILE_TYPE)

    #Detect start line by colour
    Y, X = np.where(np.all(im == START_CV2, axis = 2))
    #start = center of start line
    start=[float((X[0]+X[-1])/2),float((Y[0]+Y[-1])/2)]

    #Calculate start angle
    if X[0] != X[-1]: #If not vertical
        if (Y[0] == Y[-1]): #If horizontal
            start_angle = 90
        else:
            start_angle = math.degrees(math.atan2(X[0]-X[-1],Y[0]-Y[-1])) + DIRECTION
                
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Run Simulation For A Maximum of 100 Generations
    population.run(run_simulation, 100)

    

