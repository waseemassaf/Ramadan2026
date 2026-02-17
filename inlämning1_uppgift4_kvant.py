import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh




# Parametrar avgöringen
# -----------------------
hbar = 1.0
m = 1.0

a = 1.0
b = 3.0
V0 = 3 * np.pi**2 / (16 * a**2)

N = 800 # antal gridpunkter
x = np.linspace(0, b, N)
dx = x[1] - x[0]








# Potential V(x)
# -----------------------
V = np.zeros(N)
V[x < a] = -V0
V[x >= a] = 0.0







# Hamiltonoperator (finite difference)
# -----------------------
diag = np.ones(N) * (1.0 / dx**2)
offdiag = np.ones(N-1) * (-0.5 / dx**2)

H = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
H += np.diag(V)




# Lös egenvärdesproblemet
# -----------------------
E, psi = eigh(H)









# Plotta |psi|^2 för de fem första tillstånden
# -----------------------
plt.figure(figsize=(8,6))

for n in range(5):
    psi_n = psi[:, n]
    psi_n = psi_n / np.sqrt(np.sum(np.abs(psi_n)**2) * dx)
    
    
    plt.plot(x, np.abs(psi_n)**2 + E[n], label=f'n={n}, E={E[n]:.3f}')



plt.plot(x, V, 'k--', label='V(x)')
plt.xlabel('x')

plt.ylabel(r'$|\psi|^2 + E$')
plt.legend()



plt.title('Egenfunktioner och egenenergier')
plt.grid()
plt.show()




# Kontroll av E ≈ 0-tillstånd
# -----------------------
idx = np.argmin(np.abs(E))
print("Egenenergi närmast 0:", E[idx])

psi0 = psi[:, idx]
psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0)**2) * dx)

plt.figure()
plt.plot(x, np.abs(psi0)**2)
plt.title("Tillstånd med E ≈ 0")
plt.xlabel("x")
plt.ylabel(r"$|\psi|^2$")
plt.grid()
plt.show()
