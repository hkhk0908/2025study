import inline
import matplotlib.pyplot as plt
from qiskit import *

qr = QuantumRegister(2)
cr = QuantumRegister(2)
circuit = QuantumCircuit(qr,cr)
circuit.draw()
print(circuit.draw())
plt.show()
circuit.draw(outpout='mpl')
circuit.h(qr[0])
circuit.draw(out='mpl')