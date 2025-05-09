#!/usr/bin/python

# Convert 'requirements.txt' to 'requirements.yml' to update conda environment using:
# conda env update -f requirements.yml

import os
import re
import yaml

explicit_file = "requirements.txt" 
output_file = "requirements.yml"
env_name = "venv"

channels = set()
dependencies = []
pip_packages = []
cuda_related = []

# Match patterns
pip_hint_pattern = re.compile(r"(/pkgs/|::)?(pypi|pip)::(.+)")
conda_pkg_pattern = re.compile(r"^.+::([\w\-\+]+)=([\w\.\-]+)(=([\w\.\-]+))?$")
generic_pkg_pattern = re.compile(r"^[\w\-\.]+=[\w\.\-]+=[\w\.\-]+$")
generic_pkg_pattern_2 = re.compile(r"^[\w\-\.]+=[\w\.\-]+$")
cuda_keywords = ["cudatoolkit", "cudnn", "nccl", "cuda", "libcusparse", "libcublas", "libcufft", "libcurand"]

with open(explicit_file, "r") as f:
    for line in f:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if pip_hint_pattern.search(line.lower()): # pip packages
            pkg = pip_hint_pattern.search(line.lower()).group(3)
            pip_packages.append(pkg)
            continue

        if "::" in line: # conda::pkg=ver (e.g., conda-forge::numpy=1.25.0)
            channel, rest = line.split("::", 1)
            match = conda_pkg_pattern.match(line)
            if match:
                pkg_name = match.group(1)
                version = match.group(2)
                channels.add(channel)
                dependencies.append(f"{pkg_name}={version}")
                if any(k in pkg_name.lower() for k in cuda_keywords):
                    cuda_related.append(pkg_name)
            else:
                channels.add(channel)
                dependencies.append(rest)

        elif re.match(generic_pkg_pattern, line) or re.match(generic_pkg_pattern_2, line): # Generic build string line
            pkg_name, version = line.split("=")[:2] # Keep only name and version
            dependencies.append(f"{pkg_name}={version}")
            if any(k in pkg_name.lower() for k in cuda_keywords):
                cuda_related.append(pkg_name)

        elif line.endswith("_0") or line.endswith("_1"):
            # Possibly non-standard build string, skip it
            continue

        else: # Fallback: include the full string
            dependencies.append(line)

# Compose output dict
env_dict = {
    "name": env_name,
    "channels": sorted(channels) if channels else ["defaults"],
    "dependencies": sorted(set(dependencies))
}

# Append pip packages if any
if pip_packages:
    env_dict["dependencies"].append({
        "pip": sorted(set(pip_packages))
    })

# Write to YAML
with open(output_file, "w") as f:
    yaml.dump(env_dict, f, default_flow_style=False)

print(f"Converted environment written to {output_file}")

# CUDA/cudnn reminder
if cuda_related:
    print("Note: The environment includes CUDA-related packages:")
    for lib in cuda_related:
        print(f" - {lib}")
    print("Make sure the CUDA version is compatible with your system drivers.")