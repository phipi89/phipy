import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

import manifold3d as m3d
import numpy as np
import trimesh


def _resolve_blender_executable() -> str:
    env_value = os.environ.get("BLENDER_EXECUTABLE")
    if env_value:
        return env_value

    blender_path = shutil.which("blender")
    if blender_path:
        return blender_path

    macos_app = "/Applications/Blender.app/Contents/MacOS/Blender"
    if os.path.exists(macos_app):
        return macos_app

    raise FileNotFoundError(
        "Could not find Blender. Set BLENDER_EXECUTABLE or add `blender` to PATH."
    )


def _mesh_to_payload(mesh: trimesh.Trimesh) -> dict[str, Any]:
    mesh = mesh.copy()
    return {
        "vertices": mesh.vertices.tolist(),
        "faces": mesh.faces.tolist(),
    }


def _mesh_from_payload(payload: dict[str, Any]) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(payload["vertices"], dtype=float),
        faces=np.asarray(payload["faces"], dtype=int),
        process=False,
    )


_BLENDER_SCRIPT = r"""
import json
import sys

import bpy
import bmesh
from mathutils import bvhtree


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def object_from_payload(payload, name):
    blender_mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, blender_mesh)
    bpy.context.collection.objects.link(obj)
    blender_mesh.from_pydata(payload["vertices"], [], payload["faces"])
    blender_mesh.update()
    return obj


def triangulate_object(obj):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def object_to_payload(obj):
    triangulate_object(obj)
    return {
        "vertices": [[coord for coord in vertex.co] for vertex in obj.data.vertices],
        "faces": [list(face.vertices) for face in obj.data.polygons],
    }


def apply_remesh(obj, params):
    modifier = obj.modifiers.new(name="Remesh", type="REMESH")
    modifier.octree_depth = params["octree_depth"]
    modifier.mode = params["mode"]
    for key, value in params["options"].items():
        setattr(modifier, key, value)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return obj


def apply_boolean(obj1, obj2, params):
    obj1.hide_set(False)
    obj2.hide_set(False)
    obj1.select_set(True)
    obj2.select_set(True)
    modifier = obj1.modifiers.new(name="Boolean", type="BOOLEAN")
    modifier.operation = params["operation"]
    modifier.solver = params["solver"]
    modifier.object = obj2
    bpy.context.view_layer.objects.active = obj1
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    triangulate = obj1.modifiers.new(name="Triangulate", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=triangulate.name)
    return obj1


def apply_smooth(obj, params):
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new(name="CorrectiveSmooth", type="CORRECTIVE_SMOOTH")
    modifier.factor = params["factor"]
    modifier.iterations = params["repeat"]
    modifier.scale = params["scale"]
    modifier.smooth_type = params["smooth_type"]
    modifier.use_only_smooth = params["only_smooth"]
    modifier.use_pin_boundary = params["pin_boundaries"]
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return obj


def apply_fix_normals(obj, _params):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def apply_decimate(obj, params):
    modifier = obj.modifiers.new(name="DecimateMod", type="DECIMATE")
    modifier.ratio = params["ratio"]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return obj


def apply_extrude(obj, params):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    modifier = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
    modifier.thickness = params["thickness"]
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    triangulate = obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=triangulate.name)
    return obj


def apply_remove_verts_by_distance(obj, target_object, params):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    target_mesh = target_object.evaluated_get(depsgraph).to_mesh()
    points = [vertex.co for vertex in target_mesh.vertices]
    polygons = [polygon.vertices for polygon in target_mesh.polygons]
    bvh = bvhtree.BVHTree.FromPolygons(points, polygons, epsilon=0.0)
    target_object.to_mesh_clear()

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for vertex in bm.verts:
        vertex.select = False

    for vertex in bm.verts:
        nearest = bvh.find_nearest(obj.matrix_world @ vertex.co)
        if nearest and nearest[3] > params["threshold"]:
            vertex.select = True

    edges = {edge for vertex in bm.verts if vertex.select for edge in vertex.link_edges}
    bmesh.ops.dissolve_edges(bm, edges=list(edges), use_verts=True)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def main():
    payload_path = sys.argv[-2]
    output_path = sys.argv[-1]

    with open(payload_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    clear_scene()
    objects = [object_from_payload(mesh, f"mesh_{index}") for index, mesh in enumerate(payload["meshes"])]
    operation = payload["operation"]
    params = payload["params"]

    if operation == "remesh":
        result = apply_remesh(objects[0], params)
    elif operation == "boolean":
        result = apply_boolean(objects[0], objects[1], params)
    elif operation == "smooth":
        result = apply_smooth(objects[0], params)
    elif operation == "fix_normals":
        result = apply_fix_normals(objects[0], params)
    elif operation == "decimate":
        result = apply_decimate(objects[0], params)
    elif operation == "extrude":
        result = apply_extrude(objects[0], params)
    elif operation == "remove_verts_by_distance":
        result = apply_remove_verts_by_distance(objects[0], objects[1], params)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(object_to_payload(result), handle)


if __name__ == "__main__":
    main()
"""


def _run_blender_operation(
    operation: str,
    meshes: list[trimesh.Trimesh],
    params: dict[str, Any],
    *,
    timeout_s: float = 60.0,
) -> trimesh.Trimesh:
    """Run a mesh operation in a temporary background Blender process."""
    payload = {
        "operation": operation,
        "meshes": [_mesh_to_payload(mesh) for mesh in meshes],
        "params": params,
    }

    blender_executable = _resolve_blender_executable()

    with tempfile.TemporaryDirectory(prefix="phipy-blender-") as temp_dir:
        temp_path = Path(temp_dir)
        payload_path = temp_path / "payload.json"
        output_path = temp_path / "output.json"
        script_path = temp_path / "blender_bridge.py"

        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path.write_text(_BLENDER_SCRIPT, encoding="utf-8")

        command = [
            blender_executable,
            "--background",
            "--factory-startup",
            "--python",
            str(script_path),
            "--",
            str(payload_path),
            str(output_path),
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()

            raise TimeoutError(
                f"Blender operation '{operation}' timed out after {timeout_s:.1f}s.\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )

        if process.returncode != 0:
            raise RuntimeError(
                f"Blender subprocess failed.\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )

        if not output_path.exists():
            raise RuntimeError(
                f"Blender finished but produced no output file for operation '{operation}'.\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )

        return _mesh_from_payload(json.loads(output_path.read_text(encoding="utf-8")))


def trimesh_to_blender(mesh: trimesh.Trimesh, name: str = "mesh") -> trimesh.Trimesh:
    """Return a copy of a mesh for API symmetry with Blender-backed helpers."""
    del name
    return mesh.copy()


def blender_to_trimesh(obj: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a trimesh copy of an object produced by Blender-backed helpers."""
    return obj.copy()


def remesh_in_blender(
    mesh: trimesh.Trimesh,
    octree_depth: int = 5,
    mode: str = "SMOOTH",
    **options: Any,
) -> trimesh.Trimesh:
    """Remesh a mesh using Blender's Remesh modifier."""
    return _run_blender_operation(
        "remesh",
        [mesh],
        {
            "octree_depth": octree_depth,
            "mode": mode,
            "options": options,
        },
    )


def boolean_operation_in_blender(
    obj1: trimesh.Trimesh,
    obj2: trimesh.Trimesh,
    operation: Literal["INTERSECT", "UNION", "DIFFERENCE"],
    solver: Literal["FAST", "EXACT"] = "FAST",
) -> trimesh.Trimesh:
    """Apply a boolean operation to two meshes in Blender."""
    if operation not in ("INTERSECT", "UNION", "DIFFERENCE"):
        raise ValueError(
            f"Unknown operation {operation}, must be [INTERSECT, UNION, DIFFERENCE]."
        )

    return _run_blender_operation(
        "boolean",
        [obj1, obj2],
        {
            "operation": operation,
            "solver": solver,
        },
    )


def remesh(mesh: trimesh.Trimesh, octree_depth: int = 5) -> trimesh.Trimesh:
    """Remesh a mesh with Blender's default smooth remesh settings."""
    return remesh_in_blender(mesh, octree_depth=octree_depth)


def intersect(
    mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, **kwargs: Any
) -> trimesh.Trimesh:
    """Return the Blender boolean intersection of two meshes."""
    return boolean_operation_in_blender(mesh1, mesh2, "INTERSECT", **kwargs)


def difference(
    mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, **kwargs: Any
) -> trimesh.Trimesh:
    """Return the Blender boolean difference of two meshes."""
    return boolean_operation_in_blender(mesh1, mesh2, "DIFFERENCE", **kwargs)


def union(
    mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, **kwargs: Any
) -> trimesh.Trimesh:
    """Return the Blender boolean union of two meshes."""
    return boolean_operation_in_blender(mesh1, mesh2, "UNION", **kwargs)


def apply_corrective_smooth_modifier(
    obj: trimesh.Trimesh,
    factor: float = 1.0,
    repeat: int = 61,
    scale: float = 1.0,
    smooth_type: str = "LENGTH_WEIGHTED",
    only_smooth: bool = True,
    pin_boundaries: bool = False,
) -> trimesh.Trimesh:
    """Smooth a mesh using Blender's Corrective Smooth modifier."""
    return _run_blender_operation(
        "smooth",
        [obj],
        {
            "factor": factor,
            "repeat": repeat,
            "scale": scale,
            "smooth_type": smooth_type,
            "only_smooth": only_smooth,
            "pin_boundaries": pin_boundaries,
        },
    )


def smooth(mesh: trimesh.Trimesh, amount: int = 10, **kwargs: Any) -> trimesh.Trimesh:
    """Smooth a mesh by applying a corrective smooth modifier repeatedly."""
    return apply_corrective_smooth_modifier(mesh, repeat=amount, **kwargs)


def fix_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Recalculate mesh normals in Blender."""
    return _run_blender_operation("fix_normals", [mesh], {})


def decimate(mesh: trimesh.Trimesh, ratio: float) -> trimesh.Trimesh:
    """Reduce mesh complexity using Blender's Decimate modifier."""
    return _run_blender_operation("decimate", [mesh], {"ratio": ratio})


def extrude(mesh: trimesh.Trimesh, thickness: float) -> trimesh.Trimesh:
    """Extrude a surface mesh by a solidify thickness in Blender."""
    return _run_blender_operation("extrude", [mesh], {"thickness": thickness})


def largest_segment(
    mesh: trimesh.Trimesh,
    metric: Literal["count", "volume"] = "count",
) -> trimesh.Trimesh:
    """Return the largest connected mesh component by vertex count or volume."""
    if metric == "count":

        def key(part: trimesh.Trimesh) -> int:
            return len(part.vertices)

        watertight = False
    elif metric == "volume":

        def key(part: trimesh.Trimesh) -> float:
            return part.volume

        watertight = True
    else:
        raise ValueError(f"Unknown metric {metric}.")

    parts = trimesh.graph.split(mesh, only_watertight=watertight)
    return max(parts, key=key)


def mask_mesh(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> trimesh.Trimesh:
    """Return a mesh containing only vertices selected by a boolean mask."""
    vertices = mesh.vertices[vertex_mask]

    old_to_new_indices = np.zeros(len(mesh.vertices), dtype=int) - 1
    old_to_new_indices[vertex_mask] = np.arange(vertices.shape[0])

    faces_mask = np.all(vertex_mask[mesh.faces], axis=1)
    faces = old_to_new_indices[mesh.faces[faces_mask]]

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def remove_verts_by_distance(
    obj: trimesh.Trimesh,
    target_object: trimesh.Trimesh,
    threshold: float,
) -> trimesh.Trimesh:
    """Remove vertices farther than a threshold from a target mesh in Blender."""
    return _run_blender_operation(
        "remove_verts_by_distance",
        [obj, target_object],
        {"threshold": threshold},
    )


def make_manifold_via_m3d(input: trimesh.Trimesh) -> trimesh.Trimesh:
    """Convert a mesh to a manifold mesh using manifold3d."""
    mesh = input.copy()
    mesh.remove_unreferenced_vertices()

    vert_props = np.asarray(mesh.vertices, dtype=np.float64)
    tri_verts = np.asarray(mesh.faces, dtype=np.uint32)

    m = m3d.Mesh(
        vert_properties=vert_props,
        tri_verts=tri_verts,
    )
    m.merge()
    mani = m3d.Manifold(m)
    out = mani.to_mesh()
    verts = np.asarray(out.vert_properties)[:, :3]
    faces = np.asarray(out.tri_verts)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


def remesh_in_instant_meshes(
    mesh: trimesh.Trimesh,
    target_vertices: int = 0,
    deterministic: bool = True,
    smooth_iterations: int = 2,
    make_manifold: bool = False,
) -> trimesh.Trimesh:
    """Remesh a mesh with Instant Meshes, optionally making it manifold first."""
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as input_file:
        input_path = input_file.name
        mesh.export(input_path)

    output_path = input_path.replace(".obj", "_remeshed.obj")

    instant_meshes_path = (
        "/Applications/Instant Meshes.app/Contents/MacOS/Instant Meshes"
    )
    if not Path(instant_meshes_path).exists():
        instant_meshes_path = shutil.which("instant-meshes")

    args = [instant_meshes_path, input_path, "-S", f"{smooth_iterations}"]

    if deterministic:
        args.append("-d")

    if target_vertices:
        args += ["-v", f"{target_vertices}"]

    args += ["-o", output_path]

    subprocess.run(
        args,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    remeshed_mesh = trimesh.load_mesh(output_path)
    remeshed_mesh.merge_vertices()

    os.remove(input_path)
    os.remove(output_path)

    if make_manifold:
        remeshed_mesh = make_manifold_via_m3d(remeshed_mesh)
    return remeshed_mesh


def wireframe(mesh: trimesh.Trimesh, skip_planar: bool = True) -> np.ndarray:
    """Return edge segments for non-planar wireframe visualization."""
    edges = mesh.edges_unique
    vertices = mesh.vertices

    selected_edges = []
    for edge in edges:
        v1, v2 = edge

        connected_faces = []
        for index, face in enumerate(mesh.faces):
            if v1 in face and v2 in face:
                connected_faces.append(index)

        if len(connected_faces) == 2:
            face_normal_1 = mesh.face_normals[connected_faces[0]]
            face_normal_2 = mesh.face_normals[connected_faces[1]]

            dot_product = np.dot(face_normal_1, face_normal_2)
            angle = np.arccos(np.clip(np.abs(dot_product), -1.0, 1.0))
            angle_degrees = np.degrees(angle)

            is_coplanar = angle_degrees < 1.0 or angle_degrees > 179.0
            if is_coplanar and skip_planar:
                continue

        coords = np.vstack([vertices[v1], vertices[v2]])
        selected_edges.append(coords)

    return np.array(selected_edges)


def wireframe_vertices(mesh: trimesh.Trimesh, skip_planar: bool = True) -> np.ndarray:
    """Return wireframe vertices where multiple selected edges meet."""
    segments = wireframe(mesh, skip_planar=skip_planar)
    points = np.unique(np.concatenate(segments), axis=0)
    mask = [
        np.all(np.concatenate(segments) == point, axis=-1).sum() > 2 for point in points
    ]
    return points[mask]


def voxel_volume_mesh(mask, pitch=1.0, origin=(0.0, 0.0, 0.0)):
    """Build a jagged surface mesh from the exposed faces of a voxel mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be a 3D array, got shape {mask.shape}.")

    pitch = float(pitch)
    origin = np.asarray(origin, dtype=float).reshape(3)

    vertices = []
    faces = []
    vertex_index = {}

    face_defs = [
        ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
        ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
        ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
        ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
        ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    ]

    def get_vertex(corner):
        corner = tuple(int(v) for v in corner)
        if corner not in vertex_index:
            vertex_index[corner] = len(vertices)
            vertices.append(origin + pitch * np.asarray(corner, dtype=float))
        return vertex_index[corner]

    nx, ny, nz = mask.shape
    for i, j, k in np.argwhere(mask):
        for normal, corners in face_defs:
            ni = i + normal[0]
            nj = j + normal[1]
            nk = k + normal[2]
            neighbor_inside = (
                0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz and mask[ni, nj, nk]
            )
            if neighbor_inside:
                continue

            faces.append(
                [get_vertex((i + di, j + dj, k + dk)) for di, dj, dk in corners]
            )

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
