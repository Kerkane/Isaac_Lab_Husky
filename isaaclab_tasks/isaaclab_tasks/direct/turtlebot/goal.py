import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

##
# configuration
##

GOAL_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/goal",
    markers={
        "marker": sim_utils.SphereCfg(  # Current target (red)
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)