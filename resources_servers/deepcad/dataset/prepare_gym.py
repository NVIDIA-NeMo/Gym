# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert DeepCAD JSON or supplied views to difficulty-ranked NeMo Gym tasks."""

import argparse
import base64
import ctypes.util
import hashlib
import io
import json
import math
import mimetypes
import os
import sys
from pathlib import Path
from typing import Literal

import cadquery as cq
import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.GeomAbs import GeomAbs_Circle
from OCP.gp import gp_Vec
from PIL import Image, ImageDraw, ImageFont

from resources_servers.deepcad.dataset.augment_inputs import augment_inputs, sample_annotations


def construction_metrics(document: dict) -> dict[str, int]:
    """Count feature records and only profiles referenced by their extrusions.

    Reused profiles count once per use. Extrusions and cut/intersect operations
    count features, not the number of bodies or holes produced by a feature.
    """
    counts = {
        "extrusions": 0,
        "sketch_curves": 0,
        "additional_loops": 0,
        "cuts_and_intersections": 0,
        "arcs_and_circles": 0,
    }
    entities = document["entities"]
    for item in document["sequence"]:
        if item["type"] != "ExtrudeFeature":
            continue
        extrusion = entities[item["entity"]]
        counts["extrusions"] += 1
        counts["cuts_and_intersections"] += extrusion["operation"] in (
            "CutFeatureOperation",
            "IntersectFeatureOperation",
        )
        for reference in extrusion["profiles"]:
            profile = entities[reference["sketch"]]["profiles"][reference["profile"]]
            counts["additional_loops"] += max(0, len(profile["loops"]) - 1)
            for loop in profile["loops"]:
                curves = loop["profile_curves"]
                counts["sketch_curves"] += len(curves)
                counts["arcs_and_circles"] += sum(curve["type"] in ("Arc3D", "Circle3D") for curve in curves)
    counts["difficulty"] = (
        4 * counts["extrusions"]
        + counts["sketch_curves"]
        + 2 * counts["additional_loops"]
        + 4 * counts["cuts_and_intersections"]
    )
    return counts


def rank_split(data_root: Path, split: list[str]) -> tuple[list[dict], list[dict]]:
    """Score the entire split before applying a limit, preserving split membership."""
    ranked, failures = [], []
    for identifier in split:
        try:
            document = json.loads((data_root / "cad_json" / f"{identifier}.json").read_text())
            ranked.append({"task_id": identifier, **construction_metrics(document)})
        except (OSError, KeyError, TypeError, ValueError) as exc:
            failures.append({"task_id": identifier, "reason": str(exc), "stage": "difficulty"})
    ranked.sort(key=lambda record: (record["difficulty"], record["task_id"]))
    return ranked, failures


def xyz(point: dict) -> np.ndarray:
    return np.array([point[axis] for axis in ("x", "y", "z")], dtype=float)


def reconstruct(document: dict) -> cq.Shape:
    """Build original (unquantized) DeepCAD sketch/extrude geometry with modern OCP."""
    body = None
    entities = document["entities"]
    for item in document["sequence"]:
        if item["type"] != "ExtrudeFeature":
            continue
        extrusion = entities[item["entity"]]
        if extrusion["start_extent"]["type"] != "ProfilePlaneStartDefinition":
            raise ValueError("Unsupported start extent")
        for profile_info in extrusion["profiles"]:
            sketch = entities[profile_info["sketch"]]
            transform = sketch["transform"]
            origin, x_axis, y_axis, normal = (xyz(transform[k]) for k in ("origin", "x_axis", "y_axis", "z_axis"))

            def point(value: dict, origin=origin, x_axis=x_axis, y_axis=y_axis) -> cq.Vector:
                return cq.Vector(*(origin + value["x"] * x_axis + value["y"] * y_axis))

            wires = []
            profile = sketch["profiles"][profile_info["profile"]]
            for loop in sorted(profile["loops"], key=lambda loop: not loop["is_outer"]):
                edges = []
                for curve in loop["profile_curves"]:
                    kind = curve["type"]
                    if kind == "Line3D":
                        start, end = (
                            point(curve["start_point"]),
                            point(curve["end_point"]),
                        )
                        if (start - end).Length > 1e-10:
                            edges.append(cq.Edge.makeLine(start, end))
                    elif kind == "Circle3D":
                        edges.append(
                            cq.Edge.makeCircle(
                                abs(curve["radius"]),
                                point(curve["center_point"]),
                                cq.Vector(*normal),
                            )
                        )
                    elif kind == "Arc3D":
                        angle = (curve["start_angle"] + curve["end_angle"]) / 2
                        vector = xyz(curve["reference_vector"])[:2]
                        rotation = np.array(
                            [
                                [math.cos(angle), -math.sin(angle)],
                                [math.sin(angle), math.cos(angle)],
                            ]
                        )
                        midpoint = xyz(curve["center_point"])[:2] + rotation @ vector * curve["radius"]
                        middle = {"x": midpoint[0], "y": midpoint[1]}
                        edges.append(
                            cq.Edge.makeThreePointArc(
                                point(curve["start_point"]),
                                point(middle),
                                point(curve["end_point"]),
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported curve {kind}")
                wires.append(cq.Wire.assembleEdges(edges))
            face = cq.Face.makeFromWires(wires[0], wires[1:])
            first = extrusion["extent_one"]["distance"]["value"]
            extent_type = extrusion["extent_type"]
            second = (
                first
                if extent_type == "SymmetricFeatureExtentType"
                else (
                    extrusion["extent_two"]["distance"]["value"] if extent_type == "TwoSidesFeatureExtentType" else 0
                )
            )
            if second:
                face = face.translate(tuple(-normal * second))
            solid = cq.Shape.cast(BRepPrimAPI_MakePrism(face.wrapped, gp_Vec(*(normal * (first + second)))).Shape())
            operation = extrusion["operation"]
            if body is None:
                if operation not in ("NewBodyFeatureOperation", "JoinFeatureOperation"):
                    raise ValueError("First extrusion must create a body")
                body = solid
            elif operation in ("NewBodyFeatureOperation", "JoinFeatureOperation"):
                body = body.fuse(solid)
            elif operation == "CutFeatureOperation":
                body = body.cut(solid)
            elif operation == "IntersectFeatureOperation":
                body = body.intersect(solid)
            else:
                raise ValueError(f"Unsupported operation {operation}")
    if body is None or not body.Solids() or not body.isValid() or body.Volume() <= 0:
        raise ValueError("DeepCAD reconstruction did not produce valid solids")
    return body


def dimension_data(shape: cq.Shape, document: dict | None = None) -> dict:
    """Measure the final B-rep in source metres; labels convert distances to mm.

    Circle/arc dimensions come from surviving edges, not unused sketch curves.
    Rectangular additive feature depths are construction dimensions, not inferred
    wall thicknesses. They are only reported for axis-aligned rectangular profiles.
    """
    bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, bounds, False, False)
    lower, upper = bounds.CornerMin(), bounds.CornerMax()
    lower = np.array([lower.X(), lower.Y(), lower.Z()])
    upper = np.array([upper.X(), upper.Y(), upper.Z()])
    circles = []
    for edge in shape.Edges():
        curve = BRepAdaptor_Curve(edge.wrapped)
        if curve.GetType() != GeomAbs_Circle:
            continue
        circle = curve.Circle()
        center, normal = circle.Location(), circle.Axis().Direction()
        # Circle parameters are angles; avoid arc-length inversion on tiny STEP arcs.
        point = curve.Value((curve.FirstParameter() + curve.LastParameter()) / 2)
        circles.append(
            {
                "center": np.array([center.X(), center.Y(), center.Z()]),
                "normal": np.array([normal.X(), normal.Y(), normal.Z()]),
                "radius": circle.Radius(),
                # A point on this edge also gives a correct leader endpoint for arcs.
                "point": np.array([point.X(), point.Y(), point.Z()]),
            }
        )
    plates = []
    for item in (document or {}).get("sequence", []):
        if item["type"] != "ExtrudeFeature":
            continue
        extrusion = document["entities"][item["entity"]]
        if extrusion["operation"] not in (
            "NewBodyFeatureOperation",
            "JoinFeatureOperation",
        ):
            continue
        for reference in extrusion["profiles"]:
            sketch = document["entities"][reference["sketch"]]

            # DeepCAD JSON can mark every loop as outer. Choose the largest
            # footprint, as the viewer does, rather than trusting that flag.
            def footprint(loop):
                points = []
                for curve in loop["profile_curves"]:
                    if curve["type"] in ("Circle3D", "Arc3D"):
                        center = xyz(curve["center_point"])[:2]
                        points.extend(
                            [
                                center - abs(curve["radius"]),
                                center + abs(curve["radius"]),
                            ]
                        )
                    else:
                        points.extend([xyz(curve["start_point"])[:2], xyz(curve["end_point"])[:2]])
                return np.prod(np.ptp(points, axis=0))

            outer = max(sketch["profiles"][reference["profile"]]["loops"], key=footprint)
            curves = outer["profile_curves"]
            if len(curves) != 4 or any(c["type"] != "Line3D" for c in curves):
                continue
            vectors = [xyz(c["end_point"]) - xyz(c["start_point"]) for c in curves]
            if any(
                abs(np.dot(a, b)) > 1e-6 * np.linalg.norm(a) * np.linalg.norm(b)
                for a, b in zip(vectors, vectors[1:] + vectors[:1])
            ):
                continue
            transform = sketch["transform"]
            normal = xyz(transform["z_axis"])
            axis = int(np.argmax(abs(normal)))
            if abs(normal[axis]) < 1 - 1e-6:
                continue
            first = extrusion["extent_one"]["distance"]["value"]
            second = (
                first
                if extrusion["extent_type"] == "SymmetricFeatureExtentType"
                else extrusion["extent_two"]["distance"]["value"]
                if extrusion["extent_type"] == "TwoSidesFeatureExtentType"
                else 0
            )
            origin = xyz(transform["origin"])
            start, end = origin - normal * second, origin + normal * first
            plates.append(
                {
                    "axis": axis,
                    "start": start,
                    "end": end,
                    "depth_mm": abs(first + second) * 1000,
                }
            )
    return {
        "min": lower,
        "max": upper,
        "size_mm": (upper - lower) * 1000,
        "circles": circles,
        "plates": plates,
    }


def tessellate_for_render(shape: cq.Shape) -> tuple[list[cq.Vector], list[tuple[int, int, int]]]:
    """Keep meshing deflection above OCC precision for very small source units."""
    size = shape.BoundingBox().DiagonalLength
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Cannot render an empty or nonfinite bounding box")
    if size < 0.01:
        vertices, faces = shape.scale(1 / size).tessellate(0.002, 0.05)
        return [cq.Vector(v.x * size, v.y * size, v.z * size) for v in vertices], faces
    return shape.tessellate(size * 0.002, 0.05)


def render_views(
    shape: cq.Shape, directory: Path, *, document: dict | None = None, engineering_annotations: bool = True
) -> dict[str, Path]:
    """Render front (-Y), side (+X), top (+Z), optionally with dimensions."""
    directory.mkdir(parents=True, exist_ok=True)
    vertices, faces = tessellate_for_render(shape)
    points = np.array([v.toTuple() for v in vertices])
    triangles = points[np.array(faces)]
    dimensions = dimension_data(shape, document)
    center = (dimensions["min"] + dimensions["max"]) / 2
    scale = 420 / np.max(dimensions["max"] - dimensions["min"])
    font = ImageFont.load_default(size=18)
    small_font = ImageFont.load_default(size=16)
    views = {"front": (0, 2, 1, -1), "side": (1, 2, 0, 1), "top": (0, 1, 2, 1)}
    result = {}
    for name, (horizontal, vertical, depth, direction) in views.items():
        image = Image.new("RGB", (1280, 1280), "white")
        draw = ImageDraw.Draw(image)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
        light = np.array([0.4, -0.5, 0.75])
        shades = 145 + 65 * np.abs(normals @ light)
        order = np.argsort(direction * triangles[:, :, depth].mean(axis=1))
        for index in order:
            triangle = triangles[index]
            x = (triangle[:, horizontal] - center[horizontal]) * scale * 2 + 640
            if name == "side":
                x = 1280 - x
            y = 640 - (triangle[:, vertical] - center[vertical]) * scale * 2
            shade = int(shades[index])
            draw.polygon(list(zip(x, y)), fill=(shade, shade, shade))
        # Downsampling smooths curved silhouettes without a GUI or GPU dependency.
        image = image.resize((640, 640), Image.Resampling.LANCZOS)
        if not engineering_annotations:
            path = directory / f"{name}.png"
            image.resize((256, 256), Image.Resampling.LANCZOS).save(path)
            result[name] = path
            continue
        draw = ImageDraw.Draw(image)

        def project(point: np.ndarray, horizontal=horizontal, vertical=vertical, name=name) -> np.ndarray:
            x = (point[horizontal] - center[horizontal]) * scale + 320
            return np.array(
                [
                    640 - x if name == "side" else x,
                    320 - (point[vertical] - center[vertical]) * scale,
                ]
            )

        def label(point, text, *, rotate=False, color="#17375e", image=image):
            box = font.getbbox(text)
            tile = Image.new("RGB", (box[2] + 10, box[3] - box[1] + 8), "white")
            ImageDraw.Draw(tile).text((5, 4 - box[1]), text, font=font, fill=color)
            if rotate:
                tile = tile.rotate(90, expand=True)
            image.paste(
                tile,
                (round(point[0] - tile.width / 2), round(point[1] - tile.height / 2)),
            )

        def arrow(start, end, *, color="#17375e", both=False, draw=draw):
            start, end = np.array(start), np.array(end)
            draw.line([tuple(start), tuple(end)], fill=color, width=1)
            length = np.linalg.norm(end - start)
            if length < 1e-8:
                return
            vector = (end - start) / length
            perpendicular = np.array([-vector[1], vector[0]])
            for tip, sign in [(end, 1)] + ([(start, -1)] if both else []):
                base = tip - sign * vector * min(8, length / 3)
                draw.polygon(
                    [
                        tuple(tip),
                        tuple(base + perpendicular * 3),
                        tuple(base - perpendicular * 3),
                    ],
                    fill=color,
                )

        def value(number):
            return f"{number:.3f}".rstrip("0").rstrip(".")

        a, b = project(dimensions["min"]), project(dimensions["max"])
        left, right = sorted([a[0], b[0]])
        top, bottom = sorted([a[1], b[1]])
        draw.text(
            (20, 15),
            f"{name.upper()} VIEW  |  dimensions in mm",
            font=font,
            fill="#17375e",
        )
        for x in (left, right):
            draw.line([(x, bottom + 5), (x, 565)], fill="#17375e")
        arrow((left, 555), (right, 555), both=True)
        label(
            ((left + right) / 2, 580),
            f"{'XYZ'[horizontal]} {value(dimensions['size_mm'][horizontal])} mm",
        )
        for y in (top, bottom):
            draw.line([(right + 5, y), (565, y)], fill="#17375e")
        arrow((555, top), (555, bottom), both=True)
        label(
            (587, (top + bottom) / 2),
            f"{'XYZ'[vertical]} {value(dimensions['size_mm'][vertical])} mm",
            rotate=True,
        )

        groups = {}
        for circle in dimensions["circles"]:
            # Do not label a foreshortened circle as a true radius.
            if abs(circle["normal"][depth]) < 1 - 1e-6:
                continue
            key = round(circle["radius"] * 1000, 6)
            location = tuple(round(float(circle["center"][axis]) * 1000, 6) for axis in (horizontal, vertical))
            groups.setdefault(key, {})[location] = circle
        groups = sorted(groups.items())
        # Extra radius groups go in additional rows, keeping leaders outside the body.
        for index, (radius, locations) in enumerate(groups[:6]):
            position = (110 + (index % 3) * 180, 58 + (index // 3) * 26)
            circle = min(
                locations.values(),
                key=lambda c: np.linalg.norm(project(c["point"]) - position),
            )
            arrow(
                (position[0], position[1] + 10),
                project(circle["point"]),
                color="#7c3b00",
            )
            prefix = f"{len(locations)}x " if len(locations) > 1 else ""
            label(position, f"{prefix}R{value(radius)} mm", color="#7c3b00")
            for circle in locations.values():
                x, y = project(circle["center"])
                draw.line([(x - 4, y), (x + 4, y)], fill="#7c3b00")
                draw.line([(x, y - 4), (x, y + 4)], fill="#7c3b00")
        plates = [p for p in dimensions["plates"] if p["axis"] in (horizontal, vertical)]
        for index, plate in enumerate(plates[:2]):
            a, b = project(plate["start"]), project(plate["end"])
            if plate["axis"] == vertical:
                x = 80 - index * 25
                arrow((x, a[1]), (x, b[1]), both=True, color="#7c3b00")
                label(
                    (x - 15, (a[1] + b[1]) / 2),
                    f"depth {value(plate['depth_mm'])} mm",
                    rotate=True,
                    color="#7c3b00",
                )
            else:
                y = 100 - index * 25
                arrow((a[0], y), (b[0], y), both=True, color="#7c3b00")
                label(
                    ((a[0] + b[0]) / 2, y - 15),
                    f"depth {value(plate['depth_mm'])} mm",
                    color="#7c3b00",
                )
        draw.text(
            (20, 616),
            f"Axes: {'XYZ'[horizontal]} horizontal, {'XYZ'[vertical]} up; shared scale",
            font=small_font,
            fill="#17375e",
        )
        path = directory / f"{name}.png"
        image.resize((256, 256), Image.Resampling.LANCZOS).save(path)
        result[name] = path
    return result


def sample_view_mode(
    task_id: str, *, diagonal_view_probability: float = 0.75, seed: int = 42
) -> Literal["axis", "diagonal"]:
    """Sample once per task, independently of ordering, workers, or failed assets."""
    if not 0 <= diagonal_view_probability <= 1:
        raise ValueError("diagonal_view_probability must be between 0 and 1")
    # Use a separate hash domain from difficulty-bucket candidate selection.
    draw = int.from_bytes(hashlib.sha256(f"deepcad-views:{seed}:{task_id}".encode()).digest()[:8], "big") / 2**64
    return "diagonal" if diagonal_view_probability == 1 or draw < diagonal_view_probability else "axis"


def render_diagonal_views(shape: cq.Shape, directory: Path) -> dict[str, Path]:
    """Use BenchCAD's diagonal cameras, VTK shading and 256px composite layout.

    The shared parallel scale starts at 0.55 and expands only when a projected
    corner would clip. VTK handles visibility with a depth buffer. Finalize the
    off-screen window after each shape to release its graphics context.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    directory.mkdir(parents=True, exist_ok=True)
    vertices, faces = tessellate_for_render(shape)
    points = np.array([vertex.toTuple() for vertex in vertices])
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = np.array([0.5, 0.5, 0.5])
    points = (points - (lower + upper) / 2) / np.max(upper - lower) + center
    cameras = []
    for direction in [(-1, -1, -1), (1, 1, 1), (1, -1, 1), (-1, 1, -1)]:
        camera = np.array(direction, dtype=float) / np.sqrt(3)
        right = np.cross([0, 0, 1], camera)
        right /= np.linalg.norm(right)
        up = np.cross(camera, right)
        projected = np.stack([(points - center) @ right, (points - center) @ up], axis=1)
        cameras.append((camera, up, projected))
    half_height = max(0.55, max(np.abs(projected).max() for _, _, projected in cameras) * 1.05)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(3)
        for vertex in face:
            cells.InsertCellPoint(int(vertex))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_points)
    mesh.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.AutoOrientNormalsOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(110 / 255, 195 / 255, 192 / 255)
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.7)
    edges = vtk.vtkFeatureEdges()
    edges.SetInputConnection(normals.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOn()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOn()
    edges.SetFeatureAngle(35)
    edge_mapper = vtk.vtkPolyDataMapper()
    edge_mapper.SetInputConnection(edges.GetOutputPort())
    edge_mapper.ScalarVisibilityOff()
    edge_actor = vtk.vtkActor()
    edge_actor.SetMapper(edge_mapper)
    edge_actor.GetProperty().SetColor(0.12, 0.12, 0.12)
    edge_actor.GetProperty().SetLineWidth(1.6)
    edge_actor.GetProperty().LightingOff()
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.AddActor(actor)
    renderer.AddActor(edge_actor)
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        if not ctypes.util.find_library("OSMesa"):
            raise RuntimeError(
                "Headless diagonal rendering requires libosmesa6; install it in the preparation runtime"
            )
        window = vtk.vtkOSOpenGLRenderWindow()
    else:
        window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(256, 256)
    window.AddRenderer(renderer)
    image_filter = vtk.vtkWindowToImageFilter()
    image_filter.SetInput(window)
    composite = Image.new("RGB", (524, 524), "white")
    try:
        for index, (direction, up, _) in enumerate(cameras):
            camera = renderer.GetActiveCamera()
            camera.SetPosition(*(center + direction * 2))
            camera.SetFocalPoint(*center)
            camera.SetViewUp(*up)
            camera.ParallelProjectionOn()
            camera.SetParallelScale(half_height)
            renderer.ResetCameraClippingRange()
            window.Render()
            image_filter.Modified()
            image_filter.Update()
            data = image_filter.GetOutput()
            pixels = vtk_to_numpy(data.GetPointData().GetScalars()).reshape(256, 256, -1)
            image = Image.fromarray(np.flipud(pixels[:, :, :3]).copy())
            composite.paste(image, (4 + (index % 2) * 260, 4 + (index // 2) * 260))
    finally:
        image_filter.SetInput(None)
        renderer.RemoveAllViewProps()
        window.RemoveRenderer(renderer)
        window.Finalize()
    path = directory / "diagonal.png"
    composite.save(path)
    return {"diagonal": path}


def prepare_views(
    shape: cq.Shape,
    directory: Path,
    *,
    task_id: str,
    diagonal_view_probability: float = 0.75,
    seed: int = 42,
    document: dict | None = None,
    no_annotations_probability: float = 0.75,
) -> dict[str, Path]:
    """Render only the view combination chosen for this prepared example."""
    annotations = sample_annotations(task_id, no_annotations_probability=no_annotations_probability, seed=seed)
    if sample_view_mode(task_id, diagonal_view_probability=diagonal_view_probability, seed=seed) == "diagonal":
        return render_diagonal_views(shape, directory)
    return render_views(
        shape,
        directory,
        document=document,
        engineering_annotations=annotations,
    )


def task_row(
    task_id: str,
    reference_step: str,
    images: dict[str, Path],
    *,
    metrics: dict[str, int],
    dimensions_mm: list[float] | None = None,
    seed: int = 42,
    engineering_annotations: bool = True,
    reference_operations: list[str] | None = None,
    operation_hint_probability: float = 0.0,
) -> dict:
    """Keep the reference path in verifier metadata, never in the model prompt."""
    diagonal = set(images) == {"diagonal"}
    description = "Reproduce the geometry shown in the following views."
    if engineering_annotations and dimensions_mm is not None:
        description += (
            " Overall bounding-box dimensions in mm (X,Y,Z): "
            + ", ".join(f"{value:.6g}" for value in dimensions_mm)
            + "."
        )
    content = [{"type": "input_text", "text": description}]
    for view in ("diagonal",) if diagonal else ("front", "side", "top"):
        path = images[view]
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in ("image/png", "image/jpeg", "image/webp"):
            raise ValueError(f"Unsupported image format: {path}")
        payload = path.read_bytes()
        if not diagonal:
            # Supplied manifests follow the same 256px per-view input contract.
            with Image.open(path) as image:
                output = io.BytesIO()
                image.convert("RGBA").resize((256, 256), Image.Resampling.LANCZOS).save(output, format="PNG")
                payload = output.getvalue()
                mime = "image/png"
        content.append({"type": "input_text", "text": f"{view.capitalize()} view:"})
        content.append(
            {
                "type": "input_image",
                "detail": "high",
                "image_url": f"data:{mime};base64," + base64.b64encode(payload).decode(),
            }
        )
    row = {
        "task_source": "deepcad_simple_agent",
        "responses_create_params": {
            "input": [
                {"role": "system", "content": ""},
                {"role": "user", "content": content},
            ]
        },
        "verifier_metadata": {
            "task_id": task_id,
            "reference_step": reference_step,
            "difficulty": metrics["difficulty"],
            "view_mode": "diagonal" if diagonal else "axis",
            "engineering_annotations": engineering_annotations,
            "construction_metrics": {key: value for key, value in metrics.items() if key != "difficulty"},
        },
    }
    row = augment_inputs(row, seed=seed)
    if reference_operations is not None:
        from resources_servers.deepcad.operations import add_operation_hint

        row = add_operation_hint(row, reference_operations, probability=operation_hint_probability, seed=seed)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        type=Path,
        help="JSONL: task_id, cad_json, reference_step, images {front,side,top}",
    )
    source.add_argument(
        "--data-root",
        type=Path,
        help="Extracted DeepCAD data; reconstruct and render the chosen split",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Select easiest N valid tasks after ranking the whole split",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--diagonal-view-probability",
        type=float,
        default=0.75,
        help="Probability of four diagonal views (0: front/side/top; 1: diagonal only)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for fixed per-task view, annotation, and color choices"
    )
    parser.add_argument("--no-annotations-probability", type=float, default=0.75)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 <= args.diagonal_view_probability <= 1:
        parser.error("--diagonal-view-probability must be between 0 and 1")
    if not 0 <= args.no_annotations_probability <= 1:
        parser.error("--no-annotations-probability must be between 0 and 1")
    args.reference_root.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    if args.manifest:
        records = []
        for line in args.manifest.read_text().splitlines():
            record = json.loads(line)
            if "cad_json" not in record:
                raise ValueError(
                    "Manifest requires cad_json for construction difficulty; STEP/images cannot supply operation counts"
                )
            record["metrics"] = construction_metrics(
                json.loads((args.manifest.parent / record["cad_json"]).read_text())
            )
            records.append(record)
        records.sort(key=lambda record: (record["metrics"]["difficulty"], record["task_id"]))
        for record in records[: args.limit]:
            annotations = sample_annotations(
                record["task_id"], no_annotations_probability=args.no_annotations_probability, seed=args.seed
            )
            reference = Path(record["reference_step"])
            reference = reference if reference.is_absolute() else args.manifest.parent / reference
            shape = cq.importers.importStep(str(reference)).val()
            if not shape.isValid() or not shape.Solids() or shape.Volume() <= 0:
                raise ValueError(f"Invalid reference: {reference}")
            name = f"{len(rows):06d}.step"
            cq.exporters.export(shape, str(args.reference_root / name))
            images = {key: args.manifest.parent / value for key, value in record["images"].items()}
            if (
                sample_view_mode(
                    record["task_id"], diagonal_view_probability=args.diagonal_view_probability, seed=args.seed
                )
                == "diagonal"
            ):
                images = render_diagonal_views(shape, args.reference_root / "views" / record["task_id"])
            elif not annotations:
                images = render_views(
                    shape, args.reference_root / "views" / record["task_id"], engineering_annotations=False
                )
            rows.append(
                task_row(
                    record["task_id"],
                    name,
                    images,
                    metrics=record["metrics"],
                    dimensions_mm=dimension_data(shape)["size_mm"].tolist(),
                    seed=args.seed,
                    engineering_annotations=annotations,
                )
            )
    else:
        split = json.loads((args.data_root / "train_val_test_split.json").read_text())[args.split]
        ranked, failures = rank_split(args.data_root, split)
        args.output.with_suffix(".difficulty.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in ranked)
        )
        print(
            json.dumps(
                {
                    "ranked": len(ranked),
                    "difficulty_failures": len(failures),
                    "order": "ascending difficulty, then task_id",
                }
            ),
            flush=True,
        )
        for record in ranked:
            if len(rows) >= args.limit:
                break
            identifier = record["task_id"]
            try:
                document = json.loads((args.data_root / "cad_json" / f"{identifier}.json").read_text())
                shape = reconstruct(document)
                name = identifier.replace("/", "_") + ".step"
                cq.exporters.export(shape, str(args.reference_root / name))
                images = prepare_views(
                    shape,
                    args.reference_root / "views" / identifier,
                    task_id=identifier,
                    diagonal_view_probability=args.diagonal_view_probability,
                    seed=args.seed,
                    document=document,
                    no_annotations_probability=args.no_annotations_probability,
                )
                metrics = {key: value for key, value in record.items() if key != "task_id"}
                rows.append(
                    task_row(
                        identifier,
                        name,
                        images,
                        metrics=metrics,
                        dimensions_mm=dimension_data(shape)["size_mm"].tolist(),
                        seed=args.seed,
                        engineering_annotations=sample_annotations(
                            identifier, no_annotations_probability=args.no_annotations_probability, seed=args.seed
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - record per-asset OCP reconstruction failures
                failures.append({"task_id": identifier, "reason": str(exc)})
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {
        "written": len(rows),
        "failed": failures,
        "split": args.split if args.data_root else None,
        "ordering": "ascending difficulty, then task_id",
        "difficulty_formula": "4*extrusions + sketch_curves + 2*additional_loops + 4*cuts_and_intersections",
        "diagonal_view_probability": args.diagonal_view_probability,
        "view_seed": args.seed,
        "input_augmentation_seed": args.seed,
        "no_annotations_probability": args.no_annotations_probability,
        "annotation_counts": {
            str(value): sum(row["verifier_metadata"]["engineering_annotations"] == value for row in rows)
            for value in (False, True)
        },
        "color_counts": {
            color: sum(row["verifier_metadata"]["foreground_color"] == color for row in rows)
            for color in sorted({row["verifier_metadata"]["foreground_color"] for row in rows})
        },
        "view_counts": {
            mode: sum(row["verifier_metadata"]["view_mode"] == mode for row in rows) for mode in ("axis", "diagonal")
        },
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"written": len(rows), "failed": len(failures), "output": str(args.output)}))
    if len(rows) < args.limit:
        raise SystemExit("Not enough valid tasks; inspect the preparation report")


if __name__ == "__main__":
    main()
