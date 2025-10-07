import numpy as np
import matplotlib.pyplot as plt


# Parameters

R1 = 0.1      # Inner cylinder radius [m]
R2 = 0.3      # Outer cylinder radius [m]
omega = 10    # Angular velocity of inner cylinder [rad/s]


# Compute velocity profile

r = np.linspace(R1, R2, 200)
v_theta = (omega * R1**2 / (R2**2 - R1**2)) * ((R2**2 / r) - r)

# Compute dimensionless quantities
r_star = (r - R1) / (R2 - R1)  # normalized radial coordinate
v_star = v_theta / (omega * R1)  # normalized velocity

# Compute velocity gradient (shear rate)
dv_dr = np.gradient(v_theta, r)


# Plot 1: Velocity profile

plt.figure(figsize=(9, 6))
plt.plot(r, v_theta, 'b-', label='Velocity Profile', linewidth=2)

# Mark boundary conditions
plt.plot(R1, omega*R1, 'ro', label='$v(R_1)=\omega R_1$', markersize=8)
plt.plot(R2, 0, 'go', label='$v(R_2)=0$', markersize=8)

# Highlight flow region
plt.fill_between(r, 0, v_theta, color='lightblue', alpha=0.3, label='Flow Region')

# Annotate maximum velocity
max_idx = np.argmax(v_theta)
plt.annotate(f'Max Velocity = {v_theta[max_idx]:.2f} m/s',
             xy=(r[max_idx], v_theta[max_idx]),
             xytext=(r[max_idx]+0.02, v_theta[max_idx]-1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, color='darkred')

# Labels and layout
plt.xlabel('Radial Position, r [m]', fontsize=12)
plt.ylabel('Tangential Velocity, $v_\\theta$ [m/s]', fontsize=12)
plt.title('Velocity Profile between Coaxial Cylinders', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([R1, R2])
plt.ylim([0, np.max(v_theta)*1.2])

# Add parameter info box
param_text = f"$R_1$ = {R1} m\n$R_2$ = {R2} m\n$\\omega$ = {omega} rad/s"
plt.text(R1 + 0.005, np.max(v_theta)*0.8, param_text, 
         bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.show()


# Plot 2: Velocity Gradient (Shear Rate)

plt.figure(figsize=(8, 5))
plt.plot(r, dv_dr, 'm--', linewidth=2, label='Velocity Gradient $dv_\\theta/dr$')
plt.axhline(0, color='k', linestyle=':')
plt.xlabel('Radial Position, r [m]', fontsize=12)
plt.ylabel('Shear Rate, $dv_\\theta/dr$ [1/s]', fontsize=12)
plt.title('Shear Rate Distribution', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Plot 3: Dimensionless Velocity Profile

plt.figure(figsize=(8, 5))
plt.plot(r_star, v_star, 'c-', linewidth=2, label='Dimensionless Velocity $v^*$')
plt.xlabel('Normalized Radius $(r - R_1)/(R_2 - R_1)$', fontsize=12)
plt.ylabel('Normalized Velocity $v_\\theta / (\\omega R_1)$', fontsize=12)
plt.title('Dimensionless Velocity Profile', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
