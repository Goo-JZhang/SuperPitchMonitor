with open('C:/SuperPitchMonitor/CMakeLists.txt', 'rb') as f:
    raw = f.read(100)
    
print(f"First 100 bytes (hex): {raw.hex()}")
print(f"First 100 bytes (raw): {raw}")

# Check for line endings
if b'\r\n' in raw:
    print("\nLine endings: CRLF (Windows)")
elif b'\n' in raw:
    print("\nLine endings: LF (Unix)")
else:
    print("\nLine endings: None found in first 100 bytes")
