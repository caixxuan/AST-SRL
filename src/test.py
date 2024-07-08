import carla
import random
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

world = client.load_world('Town05')
walker_bp = world.get_blueprint_library().filter("walker.pedestrian.*")
controller_bp = world.get_blueprint_library().find('controller.ai.walker')

walker_bp = random.choice(walker_bp)
spawn_point = random.choice(world.get_map().get_spawn_points())
walker = world.spawn_actor(walker_bp, spawn_point)

walker_controller = world.spawn_actor(controller_bp, carla.Transform(), walker)

target_location = world.get_random_location_from_navigation()  # 指定目标位置的坐标
walker_controller.start()
walker_controller.go_to_location(target_location)
walker_controller.set_max_speed(1.3)  # 设置行人的最大速度 (1.3 m/s)

# 获取 spectator
spectator = world.get_spectator()
# 设置 spectator 的变换
spectator.set_transform(spawn_point)

while True:
    curr_location = walker.get_location()
    dist = curr_location.distance(target_location)
    if dist < 1.0:  # 如果距离小于1米,则认为已到达目标位置
        break
    
walker_controller.stop()
walker.destroy()
walker_controller.destroy()