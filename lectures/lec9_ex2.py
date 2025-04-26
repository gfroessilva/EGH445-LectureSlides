import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Set default font size for plots
plt.rcParams['font.size'] = 14 

# 1. Nonlinear System Dynamics
def nonlinear_system(x, u):
    # Scalar system dynamics: dx/dt = -x + x^3 + u
    return -x + x**3 + u

# 2. Linear Controller Parameters (derived above)
Ad = 0.905
Bd = 0.095
K = 4.263
Ts = 0.1
x_eq = 0.0 # Equilibrium state

# 3. Simulation Setup
t_start = 0
t_end = 1.0 # Shorter simulation time might be enough
n_steps = int(t_end / Ts)

# Initial Conditions to compare
x0_small = 2
x0_large = 2.4

# Function to run the simulation for a given x0
def run_simulation(x0_val):
    t_history = [t_start]
    x_history = [x0_val]
    u_history = []
    current_t = t_start
    current_x = np.array([x0_val]) # State needs to be array for solve_ivp

    print(f"\nRunning simulation for x(0) = {x0_val}...")
    for k in range(n_steps):
        # Calculate discrete control
        x_deviation = current_x[0] - x_eq
        u_k = -K * x_deviation
        # Optional: Limit control effort
        # u_k = np.clip(u_k, -u_max, u_max)
        u_history.append(u_k)

        # Define ODE function for this interval (constant u_k)
        def ode_interval(t, y): # y is a 1-element array
            return np.array([nonlinear_system(y[0], u_k)])

        # Simulate one interval Ts
        t_interval = (current_t, current_t + Ts)
        sol_interval = solve_ivp(
            ode_interval,
            t_interval,
            current_x,
            method='RK45',
            t_eval=[current_t + Ts] # Evaluate only at the end
        )

        if sol_interval.status != 0:
            print(f"Simulation failed at step {k} (t={current_t:.2f}) for x(0)={x0_val}")
            # Pad history for consistent plotting length if needed
            failed_steps = n_steps - k
            t_history.extend(np.linspace(current_t + Ts, t_end, failed_steps))
            x_history.extend([np.nan] * failed_steps)
            u_history.extend([np.nan] * failed_steps)
            break

        # Update state and time
        current_t += Ts
        current_x = sol_interval.y[:, -1]
        t_history.append(current_t)
        x_history.append(current_x[0]) # Store scalar state

        # Stop if state diverges excessively
        if abs(current_x[0]) > 10:
            print(f"State diverging at step {k} (t={current_t:.2f}) for x(0)={x0_val}. Stopping.")
            # Pad history
            failed_steps = n_steps - k -1 # Adjust padding index
            if failed_steps > 0:
                t_history.extend(np.linspace(current_t + Ts, t_end, failed_steps))
                x_history.extend([np.nan] * failed_steps)
                # Need to decide how to pad u_history here
                # Maybe pad u_history only up to the last successful step
                # For simplicity, we'll just plot up to failure point.
            break

    return np.array(t_history), np.array(x_history), np.array(u_history)

# Run both simulations
t_small, x_small, u_small = run_simulation(x0_small)
t_large, x_large, u_large = run_simulation(x0_large)

# 4. Plotting Results
plt.figure(figsize=(10, 8))

# Plot states
plt.subplot(2, 1, 1)
plt.plot(t_small, x_small, label=f'State x(t) (x0={x0_small})', linewidth=3, linestyle='-', marker='o')
plt.plot(t_large, x_large, label=f'State x(t) (x0={x0_large})', linewidth=3, linestyle='--', marker='x')
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.ylabel('State x')
plt.title('Scalar Nonlinear System with Linear Controller (Pole @ 0.5)')
plt.grid(True)
plt.legend()
plt.ylim(min(np.nanmin(x_small), np.nanmin(x_large), -1) - 0.5, max(np.nanmax(x_small), np.nanmax(x_large), 1) + 0.5) # Adjust ylim

# Plot control inputs
plt.subplot(2, 1, 2)
# Create piecewise constant plots for control signals
final_idx_small = len(u_small)
if final_idx_small > 0:
    t_plot_u_small = np.repeat(t_small[:final_idx_small], 2)
    u_plot_small = np.repeat(u_small[:final_idx_small], 2)
    t_plot_u_small = np.concatenate(([t_plot_u_small[0]], t_plot_u_small, [t_plot_u_small[-1]+Ts]))
    u_plot_small = np.concatenate(([u_plot_small[0]], u_plot_small, [u_plot_small[-1]]))
    plt.plot(t_plot_u_small, u_plot_small, label=f'Control u[kT] (x0={x0_small})', linewidth=3, marker='o')

final_idx_large = len(u_large)
if final_idx_large > 0:
    t_plot_u_large = np.repeat(t_large[:final_idx_large], 2)
    u_plot_large = np.repeat(u_large[:final_idx_large], 2)
    t_plot_u_large = np.concatenate(([t_plot_u_large[0]], t_plot_u_large, [t_plot_u_large[-1]+Ts]))
    u_plot_large = np.concatenate(([u_plot_large[0]], u_plot_large, [u_plot_large[-1]]))
    plt.plot(t_plot_u_large, u_plot_large, label=f'Control u[kT] (x0={x0_large})', linewidth=3, linestyle='--', marker='x')

plt.ylabel('Control Input u')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()