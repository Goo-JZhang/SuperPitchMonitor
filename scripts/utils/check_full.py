with open('C:/SuperPitchMonitor/CMakeLists.txt', 'rb') as f:
    raw = f.read()

print(f"File size: {len(raw)} bytes")
print(f"First 200 bytes (hex): {raw[:200].hex()}")

# Find first non-null byte
for i, b in enumerate(raw):
    if b != 0:
        print(f"\nFirst non-null byte at offset {i}: {hex(b)}")
        try:
            char = chr(b) if 32 <= b < 127 else '?'
            print(f"As char: {char}")
        except:
            pass
        print(f"Content from offset {i}: {raw[i:i+100]}")
        break
