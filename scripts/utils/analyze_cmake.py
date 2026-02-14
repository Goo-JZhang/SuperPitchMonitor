import re

with open('C:/SuperPitchMonitor/CMakeLists.txt', 'r') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print("\nLines containing 'Source/':")
for i, line in enumerate(lines, 1):
    if 'Source/' in line:
        print(f"  Line {i}: {line.strip()}")

print("\nLines containing '.cpp':")
for i, line in enumerate(lines, 1):
    if '.cpp' in line:
        print(f"  Line {i}: {line.strip()}")

print("\nLines containing 'target' or 'add':")
for i, line in enumerate(lines, 1):
    if 'target' in line.lower() or 'add_' in line.lower():
        print(f"  Line {i}: {line.strip()}")
