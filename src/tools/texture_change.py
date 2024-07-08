import carla,time,sys,os
import random, cv2, torch
import numpy as np

from safebench.scenario.scenario_policy.adv_patch import ObjectDetection
from safebench.scenario.scenario_definition.object_detection.vehicle import Detection_Vehicle
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider

def texture_change_(veh,image):
    object_list=list(filter(lambda k: 'SM_Tesla' in k, world.get_names_of_all_objects()))
    print(object_list)

    inputs = np.array(image.detach().cpu().numpy()*255, dtype=np.int)[0].transpose(1, 2, 0)
    height = 1024
    texture = carla.TextureColor(height,height)
    # TODO: run in multi-processing?
    for x in range(height):
        for y in range(height):
            r = int(inputs[x,y,0])
            g = int(inputs[x,y,1])
            b = int(inputs[x,y,2])
            a = int(255)
            # texture.set(x,height -0-y - 1, carla.Color(r,g,b,a))
            texture.set(height-x-1, height-y-1, carla.Color(r,g,b,a))

    for o_name in object_list:
        # print('initialize_actors: ', o_name)
        world.apply_color_texture_to_object(o_name, carla.MaterialParameter.Diffuse, texture)

def CUDA(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.cuda()

class Config:
    def __init__(self):
        self.ego_id = 0
        self.texture_dir = 'safebench/scenario/scenario_data/template_od/buaa.jpg'


client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

world = client.load_world('Town10HD')
CarlaDataProvider.set_world(world)
blueprint_library = world.get_blueprint_library()
veh_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))

spawn_point = random.choice(world.get_map().get_spawn_points())
# spawn_point = carla.Transform(carla.Location(x=-50.00, y=203.00, z=2.0))

veh = world.spawn_actor(veh_bp, spawn_point)

# 获取 spectator
spectator = world.get_spectator()
# 设置 spectator 的变换
spectator.set_transform(spawn_point)
config = Config()

agent = ObjectDetection({
        'ego_action_dim': 2, 
        'model_path': None, 
        'batch_size': 16, 
        'ROOT_DIR': '/home/oem/SafeBench/SafeBench_CaiXuan',
        'texture_dir': config.texture_dir,
        'type': 'eval'
        }, None)

agent.set_mode('eval')
ret,_ = agent.get_init_action(obs='1')

detection_vehicle = Detection_Vehicle(world, veh, config)

# 检查------------------caixuan
s = world.get_names_of_all_objects()
# 获取所有 actor
all_actors = world.get_actors()
# 过滤出车辆 actor
vehicles = []
for actor in all_actors:
    if actor.type_id.startswith('vehicle'):
        vehicles.append(actor)
# 打印车辆列表
print(f"Found {len(vehicles)} vehicles:")
for vehicle in vehicles:
    print(f"- {vehicle.type_id}")


while True:
    world.tick()
    detection_vehicle.create_behavior(ret[0])
    time.sleep(1)

veh.destroy()