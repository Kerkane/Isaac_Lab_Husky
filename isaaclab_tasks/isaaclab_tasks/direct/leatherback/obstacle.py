import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

##
# configuration
##

OBSTACLE_CFG = RigidObjectCfg(
    prim_path="/World/Object/Obstacle",
    spawn=sim_utils.ConeCfg(
        radius=0.5,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(),
)