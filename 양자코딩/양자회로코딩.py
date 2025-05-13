import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
# from qiskit.providers.aer import Aer
from qiskit.visualization import plot_histogram
# from qiskit import execute
# from qiskit_ibm_provider import IBMProvider
# from qiskit.tools.monitor import job_monitor
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
# from qiskit import execute



qr = QuantumRegister(2)   # 2개의 큐비트로 구성된 양자 레지스터 생성
cr = ClassicalRegister(2) # 2개의 비트로 구성된 고전 레지스터 생성
circuit = QuantumCircuit(qr, cr) # 주어진 양자/고전 레지스터를 사용해 양자 회로 생성
circuit.draw()
print(circuit.draw())

circuit.draw(output='mpl')
plt.show()             #이미지 형상화
print(circuit.draw())

circuit.h(qr[0])  # 첫 번째 큐비트(qr[0])에 Hadamard 게이트 적용 (중첩 상태 생성)
circuit.draw(output='mpl')
plt.show()
print(circuit.draw())

circuit.cx(qr[0],qr[1])  # # 제어 큐비트 qr[0], 타깃 큐비트 qr[1]에 CNOT 게이트 적용
circuit.draw(output='mpl')
plt.show()
print(circuit.draw())

# circuit.cx(qr[1],qr[0])  # # 제어 큐비트 qr[1], 타깃 큐비트 qr[0]에 CNOT 게이트 적용
# circuit.draw(output='mpl')
# plt.show()
# print(circuit.draw())  #반대로 걸림

circuit.measure(qr,cr) # 모든 큐비트를 대응되는 고전 비트에 측정 (qr[0] → cr[0], qr[1] → cr[1])
circuit.draw(output='mpl')
plt.show()
print(circuit.draw())

# # 고전컴퓨터회로 실행
# simulator = Aer.get_backend('qasm_simulator') # 양자 회로를 실행할 QASM 시뮬레이터 백엔드 선택 (QASM은 양자 회로를 표현하는 어셈블리 언어)
# result = execute(circuit,backend = simulator).result() # 지정한 백엔드(simulator)에서 양자 회로 실행 후 결과 반환
# plot_histogram(result.get_counts(circuit)) # 측정 결과(각 비트 조합의 발생 횟수)를 막대그래프로 시각화
# plt.show()
# print(circuit.draw())

# 양자컴퓨터 회로 실행
# provider = IBMProvider()
# print(provider.backends())  # 사용 가능한 백엔드 출력

# 사용할 수 있는 실제 양자 컴퓨터 목록 가져오기
# real_devices = provider.backends(simulator=False, operational=True)

# 가장 큐가 짧은 (덜 바쁜) 백엔드 선택
# qcomp = min(real_devices, key=lambda b: b.status().pending_jobs)
# print(f"가장 덜 바쁜 백엔드: {qcomp.name}")

# qcomp = min(
#     provider.backends(simulator=False, operational=True),
#     key=lambda b: b.status().pending_jobs
# ) # IBM 백엔드 가져오기 (이전 단계에서 이미 provider 설정함)

from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_cloud") # IBM Quantum 계정 연결
print(service.backends())  #  IBM 백엔드 목록 출력

# 가장 덜 바쁜 실제 장비 백엔드 자동 선택
real_devices = service.backends(simulator=False, operational=True)
qcomp = min(real_devices, key=lambda b: b.status().pending_jobs)
print(f"선택된 백엔드: {qcomp.name}")

# # sampler = Sampler(service=service, backend="ibmq_qasm_simulator")  # 백엔드 지정 방식 변경
#
from qiskit_ibm_runtime import Session

# 회로 실행
with Session(backend=qcomp) as session:
    sampler = Sampler(session=session)
    job = sampler.run(circuit)
    result = job.result()
    counts = result.quasi_dists[0].binary_probabilities()
    print("측정 결과:", counts)
    plot_histogram(counts)
    plt.show()

from qiskit.visualization import plot_histogram
plot_histogram({'00': 0.5, '11': 0.5})
plt.show(block=True)

# qiskit_setup.py

from qiskit_ibm_runtime import QiskitRuntimeService

# 기존 저장된 계정 정보 삭제
QiskitRuntimeService.delete_account()

# IBM Cloud API 키로 새 계정 저장 (아래 API 키를 너의 것으로 바꿔줘)
QiskitRuntimeService.save_account(
    token="여기에_너의_API_키",
    channel="ibm_cloud"
)
