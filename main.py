from itertools import product
from time import perf_counter

import numpy as np
from vispy import app, scene

rng = np.random.default_rng(42)

canvas = scene.SceneCanvas(
    title="Particle Cube | Space: Assemble / Scatter",
    size=(900, 650),
    bgcolor="#080d19",
    show=True,
    keys="interactive",
)

view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(
    distance=7,
    fov=45,
    azimuth=35,
    elevation=25,
    center=(0, 0, 0),
)

grid = np.linspace(-0.9, 0.9, 13)
u, v = np.meshgrid(grid, grid)
face_colors = ["#42a5ff", "#638cff", "#42e8cf"]
face_points = []
colors = []

for axis in range(3):
    other_axes = [index for index in range(3) if index != axis]
    for side in (-1, 1):
        face = np.empty((u.size, 3), dtype=np.float32)
        face[:, axis] = side
        face[:, other_axes[0]] = u.ravel()
        face[:, other_axes[1]] = v.ravel()
        face_points.append(face)
        colors.extend([face_colors[axis]] * len(face))

edge_points = []

for axis in range(3):
    other_axes = [index for index in range(3) if index != axis]
    for side_a, side_b in product((-1, 1), repeat=2):
        edge = np.empty((45, 3), dtype=np.float32)
        edge[:, axis] = np.linspace(-1, 1, len(edge))
        edge[:, other_axes[0]] = side_a
        edge[:, other_axes[1]] = side_b
        edge_points.append(edge)

face_positions = np.concatenate(face_points)
cube = np.concatenate([*face_points, *edge_points])
face_count = len(face_positions)

scattered = rng.uniform(-2.5, 2.5, cube.shape).astype(np.float32)
curve = rng.uniform(-0.85, 0.85, cube.shape).astype(np.float32)
delay = rng.uniform(0, 0.22, (len(cube), 1))

faces = scene.visuals.Markers(parent=view.scene)
edges = scene.visuals.Markers(parent=view.scene)

progress = 1.0
target = 1.0
last_frame = perf_counter()


def draw():
    phase = np.clip((progress - delay) / 0.78, 0.0, 1.0)
    eased = phase * phase * (3.0 - 2.0 * phase)
    positions = scattered + (cube - scattered) * eased
    positions += curve * np.sin(np.pi * eased)
    positions = positions.astype(np.float32)

    faces.set_data(
        positions[:face_count],
        face_color=colors,
        edge_width=0,
        size=3,
    )
    edges.set_data(
        positions[face_count:],
        face_color="#e3faff",
        edge_width=0,
        size=5,
    )


def animate(event):
    global progress, last_frame

    now = perf_counter()
    step = (now - last_frame) / 1.35
    last_frame = now

    if target > progress:
        progress = min(target, progress + step)
    else:
        progress = max(target, progress - step)

    draw()
    if progress == target:
        timer.stop()


@canvas.events.key_press.connect
def toggle_cube(event):
    global target, last_frame

    if event.key is not None and event.key.name == "Space":
        target = 1.0 - target
        last_frame = perf_counter()
        if not timer.running:
            timer.start()


timer = app.Timer(interval=1 / 60, connect=animate)
draw()
app.run()