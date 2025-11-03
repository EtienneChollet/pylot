import numpy as np
from pylot.util.meter import StatsMeter as CPUStatsMeter, MeterDict as CPUMeterDict
from pylot.util.meter_torch import StatsMeter, MeterDict

print("=" * 70)
print("Testing GPU StatsMeter vs CPU StatsMeter")
print("=" * 70)

# Test 1: Basic add operations
print("\nTest 1: Basic add operations")
print("-" * 70)
cpu_meter = CPUStatsMeter()
gpu_meter = StatsMeter(device='cuda')

test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
for val in test_data:
    cpu_meter.add(val)
    gpu_meter.add(val)

print(f"CPU: n={cpu_meter.n}, mean={cpu_meter.mean:.6f}, std={cpu_meter.std:.6f}")
print(f"GPU: n={gpu_meter.n.item()}, mean={gpu_meter.mean.item():.6f}, std={gpu_meter.std.item():.6f}")

assert cpu_meter.n == gpu_meter.n.item(), "Sample count mismatch"
assert np.isclose(cpu_meter.mean, gpu_meter.mean.item()), "Mean mismatch"
assert np.isclose(cpu_meter.std, gpu_meter.std.item()), "Std mismatch"
print("✓ Test 1 passed!")

# Test 2: Batch operations
print("\nTest 2: Batch operations")
print("-" * 70)
cpu_meter2 = CPUStatsMeter()
gpu_meter2 = StatsMeter(device='cuda')

batch_data = np.random.randn(1000)
cpu_meter2.addN(batch_data, batch=True)
gpu_meter2.addN(batch_data, batch=True)

print(f"CPU: n={cpu_meter2.n}, mean={cpu_meter2.mean:.6f}, std={cpu_meter2.std:.6f}")
print(f"GPU: n={gpu_meter2.n.item()}, mean={gpu_meter2.mean.item():.6f}, std={gpu_meter2.std.item():.6f}")

assert cpu_meter2.n == gpu_meter2.n.item(), "Batch: Sample count mismatch"
assert np.isclose(cpu_meter2.mean, gpu_meter2.mean.item(), rtol=1e-5), "Batch: Mean mismatch"
assert np.isclose(cpu_meter2.std, gpu_meter2.std.item(), rtol=1e-5), "Batch: Std mismatch"
print("✓ Test 2 passed!")

# Test 3: Adding two meters together
print("\nTest 3: Merging two meters")
print("-" * 70)
cpu_m1 = CPUStatsMeter()
cpu_m2 = CPUStatsMeter()
gpu_m1 = StatsMeter(device='cuda')
gpu_m2 = StatsMeter(device='cuda')

data1 = [1, 2, 3, 4, 5]
data2 = [6, 7, 8, 9, 10]

cpu_m1.addN(data1)
cpu_m2.addN(data2)
gpu_m1.addN(data1)
gpu_m2.addN(data2)

cpu_combined = cpu_m1 + cpu_m2
gpu_combined = gpu_m1 + gpu_m2

print(f"CPU combined: n={cpu_combined.n}, mean={cpu_combined.mean:.6f}, std={cpu_combined.std:.6f}")
print(f"GPU combined: n={gpu_combined.n.item()}, mean={gpu_combined.mean.item():.6f}, std={gpu_combined.std.item():.6f}")

assert cpu_combined.n == gpu_combined.n.item(), "Combined: Sample count mismatch"
assert np.isclose(cpu_combined.mean, gpu_combined.mean.item()), "Combined: Mean mismatch"
assert np.isclose(cpu_combined.std, gpu_combined.std.item()), "Combined: Std mismatch"
print("✓ Test 3 passed!")

# Test 4: Scalar addition
print("\nTest 4: Adding scalar to meter")
print("-" * 70)
cpu_m = CPUStatsMeter()
gpu_m = StatsMeter(device='cuda')

data = [1, 2, 3, 4, 5]
cpu_m.addN(data)
gpu_m.addN(data)

cpu_shifted = cpu_m + 10
gpu_shifted = gpu_m + 10

print(f"CPU shifted: mean={cpu_shifted.mean:.6f}, std={cpu_shifted.std:.6f}")
print(f"GPU shifted: mean={gpu_shifted.mean.item():.6f}, std={gpu_shifted.std.item():.6f}")

assert np.isclose(cpu_shifted.mean, gpu_shifted.mean.item()), "Shifted: Mean mismatch"
assert np.isclose(cpu_shifted.std, gpu_shifted.std.item()), "Shifted: Std mismatch"
print("✓ Test 4 passed!")

# Test 5: Scalar multiplication
print("\nTest 5: Multiplying meter by scalar")
print("-" * 70)
cpu_m = CPUStatsMeter()
gpu_m = StatsMeter(device='cuda')

data = [1, 2, 3, 4, 5]
cpu_m.addN(data)
gpu_m.addN(data)

cpu_scaled = cpu_m * 2.5
gpu_scaled = gpu_m * 2.5

print(f"CPU scaled: mean={cpu_scaled.mean:.6f}, std={cpu_scaled.std:.6f}")
print(f"GPU scaled: mean={gpu_scaled.mean.item():.6f}, std={gpu_scaled.std.item():.6f}")

assert np.isclose(cpu_scaled.mean, gpu_scaled.mean.item()), "Scaled: Mean mismatch"
assert np.isclose(cpu_scaled.std, gpu_scaled.std.item()), "Scaled: Std mismatch"
print("✓ Test 5 passed!")

# Test 6: Pop operations
print("\nTest 6: Pop operations")
print("-" * 70)
cpu_m = CPUStatsMeter()
gpu_m = StatsMeter(device='cuda')

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cpu_m.addN(data)
gpu_m.addN(data)

# Pop a few values
to_pop = [1, 2, 3]
for val in to_pop:
    cpu_m.pop(val)
    gpu_m.pop(val)

print(f"CPU after pop: n={cpu_m.n}, mean={cpu_m.mean:.6f}, std={cpu_m.std:.6f}")
print(f"GPU after pop: n={gpu_m.n.item()}, mean={gpu_m.mean.item():.6f}, std={gpu_m.std.item():.6f}")

assert cpu_m.n == gpu_m.n.item(), "Pop: Sample count mismatch"
assert np.isclose(cpu_m.mean, gpu_m.mean.item()), "Pop: Mean mismatch"
assert np.isclose(cpu_m.std, gpu_m.std.item(), rtol=1e-5), "Pop: Std mismatch"
print("✓ Test 6 passed!")

# Test 7: MeterDict
print("\nTest 7: MeterDict operations")
print("-" * 70)
cpu_dict = CPUMeterDict()
gpu_dict = MeterDict(device='cuda')

metrics = {
    'loss': [0.5, 0.4, 0.3, 0.2],
    'accuracy': [0.6, 0.7, 0.8, 0.9],
    'f1': [0.55, 0.65, 0.75, 0.85]
}

for key, values in metrics.items():
    for val in values:
        cpu_dict.add(key, val)
        gpu_dict.add(key, val)

print("\nCPU MeterDict:")
for key, meter in cpu_dict.items():
    print(f"  {key}: mean={meter.mean:.6f}, std={meter.std:.6f}")

print("\nGPU MeterDict:")
for key, meter in gpu_dict.items():
    print(f"  {key}: mean={meter.mean.item():.6f}, std={meter.std.item():.6f}")

for key in metrics.keys():
    assert np.isclose(cpu_dict[key].mean, gpu_dict[key].mean.item()), f"MeterDict: {key} mean mismatch"
    assert np.isclose(cpu_dict[key].std, gpu_dict[key].std.item()), f"MeterDict: {key} std mismatch"
print("✓ Test 7 passed!")

# Test 8: Large random data
print("\nTest 8: Large random dataset")
print("-" * 70)
cpu_m = CPUStatsMeter()
gpu_m = StatsMeter(device='cuda')

large_data = np.random.randn(10000) * 100 + 50
cpu_m.addN(large_data)
gpu_m.addN(large_data)

print(f"CPU: n={cpu_m.n}, mean={cpu_m.mean:.6f}, std={cpu_m.std:.6f}")
print(f"GPU: n={gpu_m.n.item()}, mean={gpu_m.mean.item():.6f}, std={gpu_m.std.item():.6f}")

assert cpu_m.n == gpu_m.n.item(), "Large data: Sample count mismatch"
assert np.isclose(cpu_m.mean, gpu_m.mean.item(), rtol=1e-4), "Large data: Mean mismatch"
assert np.isclose(cpu_m.std, gpu_m.std.item(), rtol=1e-4), "Large data: Std mismatch"
print("✓ Test 8 passed!")

# Test 9: from_values static method
print("\nTest 9: from_values static method")
print("-" * 70)
n, mean, std = 100, 5.5, 2.3
cpu_m = CPUStatsMeter.from_values(n, mean, std)
gpu_m = StatsMeter.from_values(n, mean, std, device='cuda')

print(f"CPU from_values: n={cpu_m.n}, mean={cpu_m.mean:.6f}, std={cpu_m.std:.6f}")
print(f"GPU from_values: n={gpu_m.n.item()}, mean={gpu_m.mean.item():.6f}, std={gpu_m.std.item():.6f}")

assert cpu_m.n == gpu_m.n.item(), "from_values: Sample count mismatch"
assert np.isclose(cpu_m.mean, gpu_m.mean.item()), "from_values: Mean mismatch"
assert np.isclose(cpu_m.std, gpu_m.std.item()), "from_values: Std mismatch"
print("✓ Test 9 passed!")

print("\n" + "=" * 70)
print("All tests passed! ✓✓✓")
print("=" * 70)