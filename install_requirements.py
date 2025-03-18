"""
Module: requirements_installer.py
Description:
    This module checks for the presence of required Python packages listed in the 
    requirements.txt file and installs any that are missing. The check is performed 
    only once per execution.
"""

import os
import sys
import subprocess

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Construct the path to requirements.txt relative to the current file.
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

# Flag to indicate whether the requirements have already been checked and installed.
requirements_checked = False

# =============================================================================
# FUNCTION: install_requirements
# =============================================================================
def install_requirements():
    """
    Installs missing packages listed in the requirements.txt file using pip.

    Executes a subprocess call to pip install the required packages, routing
    the output and error streams to the standard output and error. Provides
    feedback based on the subprocess return code.
    
    Returns:
        None
    """
    print("Installing missing packages from requirements.txt...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements_path],
        stdout=sys.stdout,
        stderr=sys.stderr,
        universal_newlines=True  # Ensures subprocess output is in text mode
    )
    if result.returncode == 0:
        print("All dependencies installed successfully.")
    else:
        print(f"Installation completed with issues. Some packages may not have been installed correctly. Return code: {result.returncode}")

# =============================================================================
# FUNCTION: check_and_install_requirements
# =============================================================================
def check_and_install_requirements():
    """
    Checks if all packages listed in requirements.txt are installed and installs any missing packages.

    This function reads the requirements.txt file, attempts to import each package,
    and collects the names of any missing packages. If any packages are missing, it calls
    install_requirements() to install them. The check is performed only once per execution.
    
    Returns:
        None

    Exits:
        If the requirements.txt file is not found, the program exits with status code 1.
    """
    global requirements_checked
    # Skip checking if requirements have already been processed.
    if requirements_checked:
        print("Requirements already checked and installed.")
        return

    try:
        with open(requirements_path, 'r') as f:
            requirements = f.readlines()
        
        missing_packages = []
        
        # Loop through each requirement and attempt to import the package.
        for requirement in requirements:
            # Extract the package name (ignoring version specifications)
            package_name = requirement.strip().split('==')[0]
            try:
                __import__(package_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            print("Some packages are missing. Installing them...")
            install_requirements()
        else:
            print("All packages are already installed.")
        
        requirements_checked = True  # Mark requirements as checked to avoid re-checking.
    
    except FileNotFoundError:
        print(f"requirements.txt file not found at {requirements_path}. Please provide the file.")
        sys.exit(1)

# =============================================================================
# EXECUTION
# =============================================================================

# Run the requirements check and installation process once at the start.
check_and_install_requirements()
print("Requirements install finished.")
