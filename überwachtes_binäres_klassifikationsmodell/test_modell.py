import numpy as np
import json
import pygame

################ Load JSON "bias_and_weights" #####################
with open("bias_and_weights.json", "r") as f:
    data = json.load(f)


weight1 = data.get("weight1", [])
bias1 = data.get("bias1", [])


weight2 = data.get("weight2", [])
bias2 = data.get("bias2", [])
##################################################

########## AI Logic #####################

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def testing(data):
    z1 = np.dot(data, weight1) + bias1
    a1 = relu(z1)

    z2 = np.dot(a1, weight2) + bias2
    a2 = sigmoid(z2)

    if a2 < 0.5:
        a2 = "Unglücklich"
    else:
        a2 = "Glücklich"

    print("Output:", a2)
#######################################


################ pygame Logic ##############################################################################

#settings
real_pixel = 1000
pixel_size = 100

pygame.init()
# Fenstergröße
width, height = real_pixel, real_pixel
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Draw a face")


# colers
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


canvas = []
def generate_2D_matrix(array,input):
    array.clear() 
    for y in range(height // pixel_size):   
        array.append([])
        for x in range(width // pixel_size): 
            array[y].append(input)
generate_2D_matrix(canvas,WHITE)

def draw_canvas():
    for y in range(len(canvas)):
        for x in range(len(canvas[0])):
            pygame.draw.rect(window, canvas[y][x], (x * pixel_size, y * pixel_size, pixel_size, pixel_size))


def mouse_press():
    if pygame.mouse.get_pressed()[0]:  # Linke Maustaste
        return [BLACK] 
    if pygame.mouse.get_pressed()[2]:  # Rechte Maustaste
        return [WHITE]
    else:
        return []

def draw_pixel(color):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    x = mouse_x // pixel_size
    y = mouse_y // pixel_size
    if 0 <= x < width // pixel_size and 0 <= y < height // pixel_size:
                canvas[y][x] = color[0]
        
for_ml = []
def convert_matrix_for_ml(canvas):
    for_ml.clear()  # vorher leeren
    for row in canvas:
        for color in row:
            if color == (255, 255, 255):
                for_ml.append(0)
            elif color == (0, 0, 0):
                for_ml.append(1)
    testing(for_ml)

def next(event):
    global next_count , happy_or_unhappy
    if event.type == pygame.KEYDOWN:  
        if event.key == pygame.K_RETURN:  
            convert_matrix_for_ml(canvas)
            generate_2D_matrix(canvas,WHITE)



running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        next(event)
        draw_canvas()

        if mouse_press() != []:
            draw_pixel(mouse_press())
       

    pygame.display.update()


pygame.quit()
###################################################################################################################################









