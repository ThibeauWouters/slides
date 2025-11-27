#!/usr/bin/env python3
"""
Script to replace all fonts in PowerPoint XML files with Montserrat
"""

import os
import re
from pathlib import Path

def replace_fonts_in_file(file_path):
    """Replace all typeface attributes with Montserrat in a single XML file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace all typeface="..." with typeface="Montserrat"
    # This regex captures typeface="anything" and replaces it
    modified_content = re.sub(
        r'typeface="[^"]*"',
        'typeface="Montserrat"',
        content
    )

    # Write back if changes were made
    if modified_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        return True
    return False

def process_directory(directory):
    """Process all XML files in the directory recursively"""
    files_changed = 0

    for xml_file in Path(directory).rglob('*.xml'):
        if replace_fonts_in_file(xml_file):
            files_changed += 1
            print(f"Modified: {xml_file}")

    return files_changed

if __name__ == "__main__":
    extracted_dir = "pptx_extracted"

    if not os.path.exists(extracted_dir):
        print(f"Error: {extracted_dir} directory not found!")
        exit(1)

    print("Starting font replacement...")
    files_changed = process_directory(extracted_dir)
    print(f"\nComplete! Modified {files_changed} files.")
