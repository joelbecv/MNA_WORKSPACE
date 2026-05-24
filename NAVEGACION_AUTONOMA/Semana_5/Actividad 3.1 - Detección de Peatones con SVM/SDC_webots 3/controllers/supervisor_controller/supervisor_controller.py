from controller import Supervisor
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
while robot.step(timestep) != -1:
    pass
