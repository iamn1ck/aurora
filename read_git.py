import subprocess

def run_git(args):
    result = subprocess.run(['git'] + args, cwd='/home/n1ck/dev/xr-dusk/dusk/extern/aurora', capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running git {' '.join(args)}: {result.stderr}")
        return None
    return result.stdout

# Get openxr_integration.cpp
cpp_content = run_git(['show', 'openxr-eyes-1.0.1:lib/webgpu/openxr_integration.cpp'])
if cpp_content:
    with open('/home/n1ck/dev/xr-dusk/dusk/extern/aurora/lib/webgpu/openxr_integration.cpp.ref', 'w') as f:
        f.write(cpp_content)
    print("Successfully retrieved openxr_integration.cpp.ref")

# Get openxr_integration.hpp
hpp_content = run_git(['show', 'openxr-eyes-1.0.1:lib/webgpu/openxr_integration.hpp'])
if hpp_content:
    with open('/home/n1ck/dev/xr-dusk/dusk/extern/aurora/lib/webgpu/openxr_integration.hpp.ref', 'w') as f:
        f.write(hpp_content)
    print("Successfully retrieved openxr_integration.hpp.ref")
