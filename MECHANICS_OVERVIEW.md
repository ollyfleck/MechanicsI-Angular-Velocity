### DISCLAIMER: This report has been generated almost entirely by artificial intelligence, with only very minor corrections applied.
https://github.com/ollyfleck/MechanicsI-Angular-Velocity

# Angular Velocity Simulation — Mechanics Overview

## Project Description

This project is a 3D interactive simulation demonstrating **angular velocity** and its relationship to **tangential velocity** and **centripetal acceleration** for a rigid body (a cube) rotating about an arbitrary axis in three-dimensional space. The central equation driving the entire simulation is:

### **v = ω × r**

Where:
- **v** — tangential velocity vector at a point on the rigid body
- **ω** (omega) — angular velocity vector (magnitude = angular speed in rad/s, direction is parallel to the instantaneous axis of rotation)
- **r** — position vector from the rotation center to the point of interest

---

## Core Mechanical Concepts

### 1. Angular Velocity Vector (ω)

Angular velocity in 3D is a **vector quantity** with three components: (ωₓ, ωᵧ, ωᵤ). Key mechanical properties:

- **Direction**: The angular velocity vector points along the **instantaneous axis of rotation** (following the right-hand rule — curl fingers in the direction of rotation, thumb points along ω).
- **Magnitude**: |ω| = √(ωₓ² + ωᵧ² + ωᵤ²) represents the **angular speed** in radians per second.
- **Superposition**: Multiple rotation components about different axes combine vectorially. If the cube spins about X at 10 rad/s and Y at 5 rad/s simultaneously, the net ω = (10, 5, 0) rad/s, and the actual rotation axis is along this combined vector.

### 2. Tangential Velocity (v = ω × r)

For any point on a rotating rigid body, the tangential velocity is computed via the **cross product** of angular velocity and the position vector:

**v = ω × r = (ωᵧrᵤ - ωᵤrᵧ, ωᵤrₓ - ωₓrᵤ, ωₓrᵧ - ωᵧrₓ)**

Key insights:
- **Perpendicularity**: v is always perpendicular to both ω and r (property of the cross product).
- **Magnitude**: |v| = |ω| · |r| · sin(θ), where θ is the angle between ω and r.
- **Maximum** when the point lies in the plane perpendicular to ω (θ = 90°).
- **Zero** for points lying on the rotation axis itself (θ = 0°, sin(0) = 0).
- **Linear scaling**: Doubling ω doubles v; doubling r doubles v (for points off-axis).

### 3. Centripetal Acceleration

Points on a rotating body experience **centripetal (radial) acceleration** directed toward the rotation axis:

**a_c = ω × v = ω × (ω × r)**

Using the vector triple product identity:
**a_c = ω(ω · r) - r(ω · ω) = ω(ω · r) - r|ω|²**

Key properties:
- **Always points toward the rotation axis** (not necessarily toward the origin).
- **Magnitude**: |a_c| = |ω|² · d, where d is the perpendicular distance from the point to the rotation axis.
- **Proportional to ω²**: Doubling angular speed quadruples centripetal acceleration.
- This is the acceleration that would be felt as "G-force" by an observer on the rotating body.

### 4. Rigid Body Rotation and Orientation

The cube's orientation is tracked using a **quaternion** (q_w, q_x, q_y, q_z):

- **Why quaternions?** Euler angles suffer from **gimbal lock** (loss of a degree of freedom when two rotation axes align). Quaternions avoid this entirely.
- **Integration**: At each timestep, the angular velocity updates the orientation:
  - δθ = |ω| · Δt (incremental rotation angle)
  - δq = [cos(δθ/2), (ω̂)·sin(δθ/2)] where ω̂ is the unit vector of ω
  - q_new = δq ⊗ q_current (quaternion multiplication)
- **Normalization**: After each update, the quaternion is normalized to prevent numerical drift from accumulating.
- **Orientation to matrix**: The quaternion is converted to a 3×3 rotation matrix for applying to vertex positions.

### 5. Fixed-Frame (Extrinsic) vs. Body-Fixed (Intrinsic) Rotation

This simulation uses **fixed-frame (extrinsic) rotations**:
- The angular velocity ω is defined in the **inertial/world frame**, not the body frame.
- Rotations are applied as if the coordinate axes remain fixed while the cube rotates around them.
- This is why the delta quaternion multiplies as **q_δ ⊗ q_current** (post-multiplication would give body-fixed/intrinsic rotations).

### 6. Angular Drag and Damping

Real-world simulations require energy dissipation. This project implements **velocity-dependent damping**:

**effective_damping = base_damping - (speed_factor × correction)**

Where speed_factor = |ω| / ω_max creates **adaptive damping** — less damping at high speeds, more at low speeds. This prevents numerical instability when the cube nearly stops while avoiding over-damping during normal operation.

The angular velocity evolves as:
- **ω(t + Δt) = ω(t) + α·Δt** (where α is angular acceleration from keyboard input)
- **ω(t + Δt) = ω(t + Δt) · damping_factor** (energy loss per step)

### 7. Cross Product Mechanics (v = ω × r)

The cross product is the mathematical heart of the simulation. For vectors **a** and **b**:

**a × b = (aᵧbᵤ - aᵤbᵧ, aᵤbₓ - aₓbᵤ, aₓbᵧ - aᵧbₓ)**

Geometric interpretation:
- **Magnitude** = |a| · |b| · sin(θ) = area of the parallelogram spanned by a and b.
- **Direction**: Perpendicular to both a and b (right-hand rule).
- **Anticommutative**: a × b = -(b × a) — order matters critically.

In the context of this simulation, for each vertex of the cube:
1. Compute **r** (vertex position relative to cube center, which is the origin).
2. Compute **v = ω × r** — the tangential velocity at that vertex.
3. Compute **a_c = ω × v** — the centripetal acceleration at that vertex.

---

## Coordinate System

The simulation uses a **right-handed coordinate system**:
- **+X**: Right
- **+Y**: Up  
- **+Z**: Out of the screen (toward viewer)

The cube is centered at the origin with vertices at (±size, ±size, ±size).

---

## Visualization

The project renders several mechanical quantities visually:

| Toggle | Quantity | Visual Representation |
|--------|----------|----------------------|
| O | ω vector | Arrow from origin along angular velocity direction |
| V | Tangential velocity (v) | Vectors at vertices showing direction and magnitude of v = ω × r |
| C | Centripetal acceleration (a_c) | Vectors at vertices showing radial acceleration toward rotation axis |
| N | Face normals | Arrows perpendicular to each cube face |
| 1-8 | Per-vertex vectors | Individual toggles for velocity vectors at each of the 8 vertices |

---

## Physics Integration

The simulation uses a **fixed-timestep physics loop** for frame-rate-independent stability:

1. **Accumulate** real elapsed time into a physics accumulator each frame.
2. **Step** physics at fixed intervals (default: 1/120 s) regardless of display framerate.
3. **Integrate** angular velocity and orientation separately at each physics step.
4. **Render** using the latest physics state, scaled by visual multiplier for display purposes.

This separation of physics and rendering timesteps is critical for:
- **Deterministic behavior** (same physics regardless of FPS).
- **Stability** (prevents energy injection from large timesteps).
- **Correct physics at high FPS** (damping is properly scaled).