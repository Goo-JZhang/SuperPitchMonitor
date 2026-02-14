import re

with open('C:/SuperPitchMonitor/CMakeLists.txt', 'r') as f:
    content = f.read()

# Find all Source/ paths
matches = re.findall(r'Source/[^\s)"\']+', content)
print("Found source files:")
for m in matches:
    print(f"  {m}")
