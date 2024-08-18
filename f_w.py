from fipy import CellVariable, PeriodicGrid2DLeftRight, TransientTerm, DiffusionTerm, ConvectionTerm, Viewer
from fipy.tools import numerix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from fipy.solvers.scipy import LinearLUSolver  # Importing a solver with preconditioning capabilities

# Grid and simulation parameters
Lx = 0.03  # Length in x-direction (m)
Ly = 0.02  # Length in y-direction (m)
nx = 200  # Number of grid points in x-direction
ny = 200  # Number of grid points in y-direction
omega = 2 * np.pi * 20  # Adjusted frequency of oscillation (Hz) to be lower for visible waves
a = 0.004  # Increased amplitude of oscillation (m) to induce wave formation
initial_dt = 0.0001  # Reduced timestep for higher resolution in time
steps = 1000  # Number of timesteps to take
t = 0  # Initial time

# Mesh creation (non-periodic for closed vessel)
mesh = PeriodicGrid2DLeftRight(dx=Lx / nx, dy=Ly / ny, nx=nx, ny=ny)

# Define the regions: Fluid and Air with a slight perturbation
h = 0.01  # Mean height of the fluid layer (m)
perturbation = 0.0005 * numerix.sin(2 * np.pi * mesh.cellCenters[0] / Lx)  # Initial perturbation in fluid height
region_2 = mesh.cellCenters[1] < (h + perturbation)  # Applying perturbation
region_1 = mesh.cellCenters[1] >= (h + perturbation)

# Physical properties
rho_2 = 1070.0  # Density of the fluid (water)
mu_2 = 10  # Reduced viscosity of the fluid for clearer wave formation
rho_1 = 1.225  # Density of the air
mu_1 = 1.81e-5  # Viscosity of the air
sigma = 0.0728  # Surface tension coefficient (N/m) for water

# Cell Variables
vx = CellVariable(name="vx", mesh=mesh, value=0.0, hasOld=True)
vy = CellVariable(name="vy", mesh=mesh, value=0.0, hasOld=True)
v = CellVariable(name="v", mesh=mesh, rank=1, value=(vx, vy), hasOld=True)
g_eff = CellVariable(name="g_eff", mesh=mesh, value=9.8)  # Effective gravity (m/s^2)

# VOF, Density, Pressure, and Viscosity fields
F = CellVariable(name="F", mesh=mesh, value=0.0, hasOld=True)
F.setValue(1.0, where=region_2)
p = CellVariable(name="pressure", mesh=mesh, value=0.0)
rho = CellVariable(name="density", mesh=mesh, value=rho_1)
rho.setValue(rho_2, where=region_2)
mu = CellVariable(name="viscosity", mesh=mesh, value=mu_1)
mu.setValue(mu_2, where=region_2)

# Surface tension forces - Curvature Calculation with Boundary-Aware Adjustment
def boundary_aware_curvature(F):
    dFdx = F.grad[0]
    dFdy = F.grad[1]
    norm_grad_F = numerix.sqrt(dFdx**2 + dFdy**2)
    norm_grad_F = numerix.where(norm_grad_F > 1e-10, norm_grad_F, 1e-10)
    
    # Calculate curvature
    curvature = -(dFdx.grad[0] + dFdy.grad[1]) / norm_grad_F
    
    # Apply boundary factor only near the exterior cells
    boundary_mask = numerix.where(mesh.cellCenters[0] <= mesh.dx, 0.2, 1.0)
    boundary_mask *= numerix.where(mesh.cellCenters[0] >= (Lx - mesh.dx), 0.2, 1.0)
    adjusted_curvature = curvature * boundary_mask
    return adjusted_curvature

curv = CellVariable(name="curvature", mesh=mesh, value=boundary_aware_curvature(F))

# Navier-Stokes Equations (Projection Method)
vx_eq_predictor = (TransientTerm(coeff=rho)
                   + ConvectionTerm(coeff=rho * v)
                   == DiffusionTerm(coeff=mu)
                   + sigma * curv * F.grad[0])

vy_eq_predictor = (TransientTerm(coeff=rho)
                   + ConvectionTerm(coeff=rho * v)
                   == DiffusionTerm(coeff=mu)
                   - rho * g_eff
                   + sigma * curv * F.grad[1])

# Poisson equation for pressure correction with preconditioning
div_v = vx.grad[0] + vy.grad[1]
pressureCorrectionEq = DiffusionTerm(coeff=1.0) == div_v / initial_dt
pressure_solver = LinearLUSolver(precon=None)  # Using LU decomposition solver, you can add a preconditioner if needed

# Velocity correction
vx_correction = -p.grad[0] / rho * initial_dt
vy_correction = -p.grad[1] / rho * initial_dt

# VOF transport equation
F_eq = TransientTerm() + ConvectionTerm(coeff=v) == 0

# Boundary Conditions: No-slip walls
vx.constrain(0.0, where=mesh.exteriorFaces)
vy.constrain(0.0, where=mesh.exteriorFaces)

# Convert mesh cell centers to NumPy arrays for visualization
x_centers = numerix.array(mesh.cellCenters[0]).reshape(nx, ny)
y_centers = numerix.array(mesh.cellCenters[1]).reshape(nx, ny)

# Visualization setup
fig, axs = plt.subplots(2, 2, figsize=(20, 10))
plt.ion()

def plot_state(step):
    for ax in axs.flatten():
        ax.clear()

    # vx plot with PowerNorm
    vx_array = numerix.array(vx.value).reshape(nx, ny)
    axs[0, 0].contourf(x_centers, y_centers, vx_array, levels=70, cmap='plasma', norm=PowerNorm(gamma=0.7))
    axs[0, 0].set_title('vx')

    # vy plot
    vy_array = numerix.array(vy.value).reshape(nx, ny)
    axs[0, 1].contourf(x_centers, y_centers, vy_array, levels=70, cmap='viridis')
    axs[0, 1].set_title('vy')

    # Pressure plot
    p_array = numerix.array(p.value).reshape(nx, ny)
    axs[1, 0].contourf(x_centers, y_centers, p_array, levels=70, cmap='coolwarm')
    axs[1, 0].set_title('Pressure')

    # Fluid Fraction F plot with PowerNorm
    F_array = numerix.array(F.value).reshape(nx, ny) * 10
    axs[1, 1].contourf(x_centers, y_centers, F_array, levels=30, cmap='viridis', norm=PowerNorm(gamma=1.2))
    axs[1, 1].set_title('Fluid Fraction F')

    plt.tight_layout()
    plt.suptitle(f'Step: {step}, Time: {t:.4f} s')
    plt.pause(0.01 if step >= 5 else 1)

# Time-stepping loop
for step in range(1, steps + 1):
    # Adaptive time step based on CFL condition
    max_velocity = max((vx.value**2 + vy.value**2)**0.5)
    dt = min(initial_dt, 0.5 * min(Lx / nx, Ly / ny) / (max_velocity + 1e-12))

    # Update effective gravity with smoothing
    g_eff.setValue(9.8 + a * omega**2 * numerix.sin(omega * t))

    # Solve momentum predictor equations
    vx_eq_predictor.solve(var=vx, dt=dt)
    vy_eq_predictor.solve(var=vy, dt=dt)

    # Solve Poisson equation for pressure correction with a solver that allows preconditioning
    pressureCorrectionEq.solve(var=p, solver=pressure_solver)

    # Correct velocities
    vx.setValue(vx + vx_correction)
    vy.setValue(vy + vy_correction)
    v.setValue((vx, vy))

    # Solve VOF equation with slope limiting (if applicable)
    F_eq.solve(var=F, dt=dt)

    # Update physical properties
    rho.setValue(rho_1 + (rho_2 - rho_1) * F)
    mu.setValue(mu_1 + (mu_2 - mu_1) * F)

    # Update curvature
    curv.setValue(boundary_aware_curvature(F))

    # Visualization every step to capture dynamic wave formation
    if step % 1 == 0:
        plot_state(step)

    # Update time
    t += dt

    print(f"Step {step}/{steps} completed")

plt.ioff()
plt.show()
print("Simulation complete.")
