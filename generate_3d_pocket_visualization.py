#!/usr/bin/env python3
"""
Generate enhanced 3D visualization of peptide in receptor pocket
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches

def generate_3d_pocket_visualization(sequence, pocket_name, key_residues, output_file):
    """Generate 3D visualization of peptide in receptor pocket"""
    print(f"Generating 3D pocket visualization for {sequence} in {pocket_name}...")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Amino acid properties for coloring
    aa_colors = {
        'E': '#FF6B6B',  # Acidic - Red
        'D': '#FF6B6B',  # Acidic - Red
        'K': '#4ECDC4',  # Basic - Cyan
        'R': '#4ECDC4',  # Basic - Cyan
        'H': '#95E1D3',  # Basic - Light cyan
        'S': '#FFA07A',  # Polar - Orange
        'T': '#FFA07A',  # Polar - Orange
        'N': '#FFA07A',  # Polar - Orange
        'Q': '#FFA07A',  # Polar - Orange
        'Y': '#FFD93D',  # Aromatic - Yellow
        'F': '#FFD93D',  # Aromatic - Yellow
        'W': '#FFD93D',  # Aromatic - Yellow
        'G': '#E0E0E0',  # Special - Gray
        'P': '#E0E0E0',  # Special - Gray
        'A': '#B8B8B8',  # Hydrophobic - Dark gray
        'V': '#B8B8B8',  # Hydrophobic
        'L': '#B8B8B8',  # Hydrophobic
        'I': '#B8B8B8',  # Hydrophobic
        'M': '#B8B8B8',  # Hydrophobic
        'C': '#F9ED69',  # Sulfur - Light yellow
    }
    
    # Draw peptide backbone (extended conformation along x-axis)
    peptide_coords = []
    for i, aa in enumerate(sequence):
        x = i * 3.0
        y = 0.0
        z = 0.0
        peptide_coords.append((x, y, z, aa))
        
        # Draw amino acid sphere
        color = aa_colors.get(aa, '#CCCCCC')
        ax.scatter([x], [y], [z], c=[color], s=800, alpha=0.9, edgecolors='black', linewidth=2)
        ax.text(x, y, z + 1.5, aa, fontsize=14, fontweight='bold', ha='center', va='center')
    
    # Draw peptide backbone bonds
    for i in range(len(sequence) - 1):
        x1, y1, z1, _ = peptide_coords[i]
        x2, y2, z2, _ = peptide_coords[i + 1]
        ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', linewidth=4, alpha=0.7)
    
    # Draw receptor pocket residues around peptide
    n_residues = len(key_residues)
    pocket_center_x = (len(sequence) - 1) * 1.5
    pocket_center_y = 0
    pocket_center_z = 0
    
    # Arrange receptor residues in a sphere around peptide
    radius = 8.0
    
    for i, (residue, info) in enumerate(key_residues.items()):
        # Distribute residues around peptide in 3D
        phi = np.pi * (i / n_residues)  # Polar angle
        theta = 2 * np.pi * (i / n_residues)  # Azimuthal angle
        
        x = pocket_center_x + radius * np.sin(phi) * np.cos(theta)
        y = pocket_center_y + radius * np.sin(phi) * np.sin(theta)
        z = pocket_center_z + radius * np.cos(phi)
        
        # Draw receptor residue
        interaction_type = info.get('type', 'Unknown')
        
        if 'H-bond' in interaction_type:
            color = '#87CEEB'  # Sky blue
            marker = 'o'
        elif 'Electrostatic' in interaction_type:
            color = '#FF69B4'  # Hot pink
            marker = 's'
        elif 'Hydrophobic' in interaction_type:
            color = '#90EE90'  # Light green
            marker = '^'
        else:
            color = '#D3D3D3'  # Light gray
            marker = 'o'
        
        ax.scatter([x], [y], [z], c=[color], s=400, alpha=0.8, 
                  edgecolors='darkblue', linewidth=2, marker=marker)
        ax.text(x, y, z, residue, fontsize=8, fontweight='bold', ha='center', va='center')
        
        # Draw interaction lines to closest peptide residue
        min_dist = float('inf')
        closest_peptide = None
        for px, py, pz, paa in peptide_coords:
            dist = np.sqrt((x - px)**2 + (y - py)**2 + (z - pz)**2)
            if dist < min_dist:
                min_dist = dist
                closest_peptide = (px, py, pz)
        
        if closest_peptide:
            px, py, pz = closest_peptide
            
            if 'H-bond' in interaction_type:
                line_style = '--'
                line_color = 'blue'
                line_width = 1.5
            elif 'Electrostatic' in interaction_type:
                line_style = '-'
                line_color = 'red'
                line_width = 2
            elif 'Hydrophobic' in interaction_type:
                line_style = ':'
                line_color = 'green'
                line_width = 1.5
            else:
                line_style = '-'
                line_color = 'gray'
                line_width = 1
            
            ax.plot([px, x], [py, y], [pz, z], 
                   linestyle=line_style, color=line_color, linewidth=line_width, alpha=0.5)
    
    # Draw pocket boundary (transparent sphere)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = pocket_center_x + radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = pocket_center_y + radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = pocket_center_z + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')
    
    # Set labels and title
    ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (Å)', fontsize=12, fontweight='bold')
    ax.set_title(f'{pocket_name} Pocket with Peptide {sequence}\n3D Molecular Visualization', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Set axis limits
    ax.set_xlim(pocket_center_x - radius - 2, pocket_center_x + radius + 2)
    ax.set_ylim(pocket_center_y - radius - 2, pocket_center_y + radius + 2)
    ax.set_zlim(pocket_center_z - radius - 2, pocket_center_z + radius + 2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"3D pocket visualization saved to {output_file}")
    plt.close()

def generate_pocket_cross_section(sequence, pocket_name, output_file):
    """Generate cross-section view of peptide in pocket"""
    print(f"Generating pocket cross-section for {sequence} in {pocket_name}...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw pocket outline (simplified as ellipse)
    if pocket_name == "T1R1":
        pocket_width = 6
        pocket_height = 4
        pocket_color = '#FFE5B4'  # Peach
    else:  # T1R3
        pocket_width = 8
        pocket_height = 5.5
        pocket_color = '#E0F7FA'  # Light cyan
    
    # Draw pocket
    pocket = plt.Circle((7, 5), pocket_width/2, color=pocket_color, alpha=0.3, linewidth=3, 
                       edgecolor='darkblue', linestyle='--')
    ax.add_patch(pocket)
    
    # Add pocket label
    ax.text(7, 5 + pocket_height/2 + 1, f'{pocket_name} Pocket', 
           ha='center', va='center', fontsize=16, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='darkblue', linewidth=2))
    
    # Draw peptide inside pocket
    aa_colors = {
        'E': '#FF6B6B', 'D': '#FF6B6B', 'K': '#4ECDC4', 'R': '#4ECDC4',
        'S': '#FFA07A', 'T': '#FFA07A', 'G': '#E0E0E0', 'Y': '#FFD93D',
    }
    
    # Position peptide amino acids
    peptide_y = 5
    start_x = 7 - (len(sequence) - 1) * 0.6
    
    for i, aa in enumerate(sequence):
        x = start_x + i * 1.2
        color = aa_colors.get(aa, '#CCCCCC')
        
        # Draw amino acid
        circle = plt.Circle((x, peptide_y), 0.4, color=color, ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, peptide_y, aa, ha='center', va='center', fontsize=14, fontweight='bold', zorder=4)
        
        # Draw backbone connection
        if i < len(sequence) - 1:
            ax.plot([x + 0.4, x + 0.8], [peptide_y, peptide_y], 'k-', linewidth=3, zorder=2)
    
    # Add annotations
    ax.text(7, 1, f'Pocket Size: {pocket_name}', ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
    
    if pocket_name == "T1R1":
        size_text = "~1,070 Å³\nPrefers smaller peptides (2-4 aa)"
    else:
        size_text = "~2,374 Å³\nPrefers larger peptides (5-10 aa)"
    
    ax.text(7, 0.2, size_text, ha='center', va='center', fontsize=10, style='italic')
    
    # Add N and C terminus labels
    ax.text(start_x - 0.8, peptide_y, 'N', ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(start_x + (len(sequence) - 1) * 1.2 + 0.8, peptide_y, 'C', ha='center', va='center', 
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', edgecolor='black', linewidth=2))
    
    # Set axis properties
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.text(7, 10, f'Cross-Section View: {sequence} in {pocket_name} Pocket', 
           ha='center', va='center', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Pocket cross-section saved to {output_file}")
    plt.close()

# Main execution
if __name__ == "__main__":
    sequence = "EDGET"
    
    # T1R1 key residues
    t1r1_residues = {
        'Y220': {'type': 'H-bond'},
        'E301': {'type': 'Electrostatic'},
        'L305': {'type': 'Hydrophobic'},
        'S306': {'type': 'H-bond'},
        'S385': {'type': 'H-bond'},
        'N388': {'type': 'H-bond'},
        'D147': {'type': 'Electrostatic'},
        'Y169': {'type': 'H-bond'},
        'L75': {'type': 'Hydrophobic'},
        'A302': {'type': 'H-bond'},
    }
    
    # T1R3 key residues
    t1r3_residues = {
        "S146'": {'type': 'H-bond'},
        "S147'": {'type': 'H-bond'},
        "E148'": {'type': 'Electrostatic'},
        "T167'": {'type': 'H-bond/Hydrophobic'},
        "G168'": {'type': 'Flexibility'},
        "S170'": {'type': 'H-bond'},
        "M171'": {'type': 'Hydrophobic'},
        "D190'": {'type': 'Electrostatic'},
        "N386'": {'type': 'H-bond'},
        "Q389'": {'type': 'H-bond'},
        "W72'": {'type': 'Hydrophobic'},
        "H145'": {'type': 'H-bond'},
    }
    
    # Generate 3D visualizations
    generate_3d_pocket_visualization(sequence, "T1R1", t1r1_residues,
                                    "/home/ubuntu/peptide_3d_t1r1_pocket.png")
    generate_3d_pocket_visualization(sequence, "T1R3", t1r3_residues,
                                    "/home/ubuntu/peptide_3d_t1r3_pocket.png")
    
    # Generate cross-section views
    generate_pocket_cross_section(sequence, "T1R1", "/home/ubuntu/peptide_t1r1_cross_section.png")
    generate_pocket_cross_section(sequence, "T1R3", "/home/ubuntu/peptide_t1r3_cross_section.png")
    
    print("\nAll 3D visualizations generated successfully!")
    print("Files created:")
    print("  - peptide_3d_t1r1_pocket.png")
    print("  - peptide_3d_t1r3_pocket.png")
    print("  - peptide_t1r1_cross_section.png")
    print("  - peptide_t1r3_cross_section.png")

