"""my_controller_wall_follower controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

max_speed = 6.28

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

#Motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

#proximity sensors
prox_sensors = []
for ind in range(8):
    sensor_name = 'ps' + str(ind)
    prox_sensors.append(robot.getDevice(sensor_name))
    prox_sensors[ind].enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    for ind in range(5,8):
        print("ind:{}, val:{}".format(ind, prox_sensors[ind].getValue()))
        
    #process sensor data
    left_wall = prox_sensors[5].getValue() > 75
    front_wall = prox_sensors[7].getValue() > 75
    left_corner = prox_sensors[6].getValue() > 75
    
    left_speed = max_speed
    right_speed = max_speed
    
    if front_wall:
        print("Turning right")
        left_speed = max_speed
        right_speed = -max_speed
        
    elif left_wall:
        print("Driving forward")
        left_speed = max_speed
        right_speed = max_speed
        
    elif left_corner:
        print("Too close to wall, driving right")
        left_speed = max_speed
        right_speed = max_speed/8
        
    else:
        print("Turning left")
        left_speed = max_speed/8
        right_speed = max_speed
                
    leftMotor.setVelocity(left_speed)
    rightMotor.setVelocity(right_speed)


# Enter here exit cleanup code.
