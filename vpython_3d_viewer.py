from vpython import *

class Point:
    def __init__(self, x, y, z):
        self.coord = vector(x, y, z)
        self.links = []  # List of tuples (other_point_index, diameter)
        

class Link:
    def __init__(self, p1_index, p2_index, diameter):
        self.p1_index = p1_index
        self.p2_index = p2_index
        self.diameter = diameter

class Individual:
    def __init__(self, points, links):
        self.points = points  # List of Point objects
        self.links = links  # List of Link objects

def create_tube(start, end, radius=0.05, color=color.white):
    axis_vector = vector(end.x - start.x, end.y - start.y, end.z - start.z)
    cylinder(pos=start, axis=axis_vector, radius=radius, color=color)

def handle_keypress(evt):
    # Access the camera position
    cam_pos = scene.camera.pos
    move_speed = 0.1

    if evt.key == 'q':
        quit
    if evt.key == 'left':  # Pan left
        scene.camera.pos = vector(cam_pos.x - move_speed, cam_pos.y, cam_pos.z)
    elif evt.key == 'right':  # Pan right
        scene.camera.pos = vector(cam_pos.x + move_speed, cam_pos.y, cam_pos.z)
    elif evt.key == 'up':  # Pan up
        scene.camera.pos = vector(cam_pos.x, cam_pos.y + move_speed, cam_pos.z)
    elif evt.key == 'down':  # Pan down
        scene.camera.pos = vector(cam_pos.x, cam_pos.y - move_speed, cam_pos.z)
    else:
        pass

def visualize_individual(individual):
    scene = canvas(title="3D Chassis Visualizer", width=800, height=600)
    scene.bind('keydown', handle_keypress)
    scene.userpan = True 
    scene.autoscale = True
    
    for link in individual.links:
        start = individual.points[link.p1_index].coord
        end = individual.points[link.p2_index].coord
        create_tube(start, end, radius=link.diameter / 2)
    
    while True:
        rate(60)
    
def visualize_individuals(individuals):
    scenes = []
    

    
    for i in range(0,len(individuals),round(len(individuals)/5)):
        individual = individuals[i]
        scene = canvas(title=("3D Chassis Visualizer gen " + str(i)), width=800, height=600)
        scene.bind('keydown', lambda evt: exit() if evt.key == 'q' else None)

        for link in individual.links:
            start = individual.points[link.p1_index].coord
            end = individual.points[link.p2_index].coord
            create_tube(start, end, radius=link.diameter / 2)
        i = i +1
    scene = canvas(title=("3D Chassis Visualizer gen " + str(len(individuals))), width=800, height=600)
    scene.bind('keydown', lambda evt: exit() if evt.key == 'q' else None)
    individual = individuals[-1]
    for link in individual.links:
        start = individual.points[link.p1_index].coord
        end = individual.points[link.p2_index].coord
        create_tube(start, end, radius=link.diameter / 2)
    while True:
        rate(60)

# Example usage
if __name__ == "__main__":
    points = [Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0)]
    links = [Link(0, 1, 0.1), Link(1, 2, 0.1), Link(2, 3, 0.1), Link(3, 0, 0.1)]
    individual = Individual(points, links)
    visualize_individual(individual)
