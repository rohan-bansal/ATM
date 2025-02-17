import mujoco
import trimesh

# Load MJCF model
model = mujoco.MjModel.from_xml_path("/home/terra/dev/Research/rl2/atm/ATM/third_party/robosuite/robosuite/models/assets/objects/t_shape-visual.xml")
data = mujoco.MjData(model)

# Extract geometry
meshes = []
for i in range(model.ngeom):
    geom_type = model.geom_type[i]
    geom_size = model.geom_size[i]
    geom_pos = model.geom_pos[i]

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        size = geom_size[:3]  # Box dimensions
        mesh = trimesh.creation.box(extents=[2 * s for s in size])
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=geom_size[0])
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=geom_size[0], height=geom_size[1] * 2)
    else:
        continue  # Skip unsupported types

    mesh.apply_translation(geom_pos)
    meshes.append(mesh)

# Combine and export as STL
combined = trimesh.util.concatenate(meshes)
combined.export("output.stl")