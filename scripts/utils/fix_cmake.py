with open('C:/SuperPitchMonitor/CMakeLists.txt', 'r', newline='') as f:
    content = f.read()

# Replace LF with CRLF for Windows display
lines = content.split('\n')
print(f"Total lines: {len(lines)}")

# Find lines with Source/
print("\nLines containing 'Source/':")
for i, line in enumerate(lines, 1):
    if 'Source/' in line:
        print(f"  Line {i}: {line.strip()}")

# Find lines with target_sources or similar
print("\nLines containing 'target':")
for i, line in enumerate(lines, 1):
    if 'target' in line.lower():
        print(f"  Line {i}: {line.strip()}")
