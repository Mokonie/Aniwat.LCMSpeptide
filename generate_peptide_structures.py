#!/usr/bin/env python3
"""
Generate 2D and 3D molecular structures of peptides
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Amino acid SMILES (simplified, without stereochemistry for visualization)
AA_SMILES = {
    'A': 'C[C@H](N)C(=O)O',  # Alanine
    'R': 'NC(CCCNC(N)=N)C(=O)O',  # Arginine
    'N': 'NC(CC(N)=O)C(=O)O',  # Asparagine
    'D': 'NC(CC(=O)O)C(=O)O',  # Aspartic acid
    'C': 'NC(CS)C(=O)O',  # Cysteine
    'Q': 'NC(CCC(N)=O)C(=O)O',  # Glutamine
    'E': 'NC(CCC(=O)O)C(=O)O',  # Glutamic acid
    'G': 'NCC(=O)O',  # Glycine
    'H': 'NC(CC1=CNC=N1)C(=O)O',  # Histidine
    'I': 'CC[C@H](C)[C@H](N)C(=O)O',  # Isoleucine
    'L': 'CC(C)C[C@H](N)C(=O)O',  # Leucine
    'K': 'NCCCC[C@H](N)C(=O)O',  # Lysine
    'M': 'CSCC[C@H](N)C(=O)O',  # Methionine
    'F': 'NC(Cc1ccccc1)C(=O)O',  # Phenylalanine
    'P': 'O=C(O)[C@@H]1CCCN1',  # Proline
    'S': 'NC(CO)C(=O)O',  # Serine
    'T': 'C[C@@H](O)[C@H](N)C(=O)O',  # Threonine
    'W': 'NC(CC1=CNc2ccccc12)C(=O)O',  # Tryptophan
    'Y': 'NC(Cc1ccc(O)cc1)C(=O)O',  # Tyrosine
    'V': 'CC(C)[C@H](N)C(=O)O',  # Valine
}

def peptide_to_smiles(sequence):
    """Convert peptide sequence to SMILES string"""
    # Simplified: just concatenate amino acids
    # For proper peptide SMILES, we need to form peptide bonds
    # This is a simplified version for visualization
    
    # Build peptide SMILES manually
    if len(sequence) == 0:
        return None
    
    # Start with first amino acid
    smiles_parts = []
    for i, aa in enumerate(sequence):
        if aa not in AA_SMILES:
            return None
        smiles_parts.append(AA_SMILES[aa])
    
    # For simplicity, create a linear peptide representation
    # In reality, peptide bonds would be formed properly
    peptide_smiles = '.'.join(smiles_parts)
    
    return peptide_smiles

def generate_2d_structure(sequence, output_file):
    """Generate 2D structure of peptide"""
    print(f"Generating 2D structure for {sequence}...")
    
    # Create a simple 2D representation
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, len(sequence) + 1)
    ax.set_ylim(-2, 3)
    ax.axis('off')
    
    # Amino acid properties for coloring
    aa_properties = {
        'E': {'color': '#FF6B6B', 'type': 'Acidic', 'charge': '-'},
        'D': {'color': '#FF6B6B', 'type': 'Acidic', 'charge': '-'},
        'K': {'color': '#4ECDC4', 'type': 'Basic', 'charge': '+'},
        'R': {'color': '#4ECDC4', 'type': 'Basic', 'charge': '+'},
        'H': {'color': '#95E1D3', 'type': 'Basic', 'charge': '+'},
        'S': {'color': '#FFA07A', 'type': 'Polar', 'charge': '0'},
        'T': {'color': '#FFA07A', 'type': 'Polar', 'charge': '0'},
        'N': {'color': '#FFA07A', 'type': 'Polar', 'charge': '0'},
        'Q': {'color': '#FFA07A', 'type': 'Polar', 'charge': '0'},
        'Y': {'color': '#FFD93D', 'type': 'Aromatic', 'charge': '0'},
        'F': {'color': '#FFD93D', 'type': 'Aromatic', 'charge': '0'},
        'W': {'color': '#FFD93D', 'type': 'Aromatic', 'charge': '0'},
        'G': {'color': '#E0E0E0', 'type': 'Special', 'charge': '0'},
        'P': {'color': '#E0E0E0', 'type': 'Special', 'charge': '0'},
        'A': {'color': '#B8B8B8', 'type': 'Hydrophobic', 'charge': '0'},
        'V': {'color': '#B8B8B8', 'type': 'Hydrophobic', 'charge': '0'},
        'L': {'color': '#B8B8B8', 'type': 'Hydrophobic', 'charge': '0'},
        'I': {'color': '#B8B8B8', 'type': 'Hydrophobic', 'charge': '0'},
        'M': {'color': '#B8B8B8', 'type': 'Hydrophobic', 'charge': '0'},
        'C': {'color': '#F9ED69', 'type': 'Sulfur', 'charge': '0'},
    }
    
    # Draw peptide backbone
    for i in range(len(sequence) - 1):
        ax.plot([i, i + 1], [0, 0], 'k-', linewidth=3, zorder=1)
    
    # Draw amino acids
    for i, aa in enumerate(sequence):
        props = aa_properties.get(aa, {'color': '#CCCCCC', 'type': 'Unknown', 'charge': '0'})
        
        # Draw circle for amino acid
        circle = Circle((i, 0), 0.35, color=props['color'], ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        
        # Add amino acid letter
        ax.text(i, 0, aa, ha='center', va='center', fontsize=20, fontweight='bold', zorder=3)
        
        # Add position number
        ax.text(i, -0.7, f"Pos {i+1}", ha='center', va='center', fontsize=10, style='italic')
        
        # Add charge indicator
        if props['charge'] != '0':
            charge_circle = Circle((i + 0.25, 0.25), 0.12, color='white', ec='black', linewidth=1.5, zorder=4)
            ax.add_patch(charge_circle)
            ax.text(i + 0.25, 0.25, props['charge'], ha='center', va='center', 
                   fontsize=12, fontweight='bold', zorder=5)
    
    # Add N-terminus and C-terminus labels
    ax.text(-0.5, 0, 'N', ha='center', va='center', fontsize=16, fontweight='bold', 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(len(sequence) - 0.5, 0, 'C', ha='center', va='center', fontsize=16, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', edgecolor='black', linewidth=2))
    
    # Add title
    ax.text(len(sequence)/2 - 0.5, 2.5, f'Peptide: {sequence}', 
           ha='center', va='center', fontsize=22, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#FF6B6B', label='Acidic (D, E)'),
        mpatches.Patch(color='#4ECDC4', label='Basic (K, R, H)'),
        mpatches.Patch(color='#FFA07A', label='Polar (S, T, N, Q)'),
        mpatches.Patch(color='#FFD93D', label='Aromatic (Y, F, W)'),
        mpatches.Patch(color='#B8B8B8', label='Hydrophobic (A, V, L, I, M)'),
        mpatches.Patch(color='#E0E0E0', label='Special (G, P)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True, 
             fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"2D structure saved to {output_file}")
    plt.close()

def generate_interaction_diagram(sequence, pocket_name, key_residues, output_file):
    """Generate 2D interaction diagram showing peptide-receptor interactions"""
    print(f"Generating interaction diagram for {sequence} with {pocket_name}...")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 10)
    ax.axis('off')
    
    # Amino acid properties
    aa_properties = {
        'E': {'color': '#FF6B6B', 'type': 'Acidic'},
        'D': {'color': '#FF6B6B', 'type': 'Acidic'},
        'K': {'color': '#4ECDC4', 'type': 'Basic'},
        'R': {'color': '#4ECDC4', 'type': 'Basic'},
        'S': {'color': '#FFA07A', 'type': 'Polar'},
        'T': {'color': '#FFA07A', 'type': 'Polar'},
        'G': {'color': '#E0E0E0', 'type': 'Special'},
        'Y': {'color': '#FFD93D', 'type': 'Aromatic'},
    }
    
    # Draw peptide in the center
    peptide_y = 5
    for i, aa in enumerate(sequence):
        x = 2 + i * 1.2
        props = aa_properties.get(aa, {'color': '#CCCCCC', 'type': 'Unknown'})
        
        circle = Circle((x, peptide_y), 0.4, color=props['color'], ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, peptide_y, aa, ha='center', va='center', fontsize=16, fontweight='bold', zorder=3)
    
    # Add peptide label
    ax.text(2 + len(sequence) * 0.6, peptide_y + 1.2, f'Peptide: {sequence}', 
           ha='center', va='center', fontsize=18, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black', linewidth=2))
    
    # Draw receptor pocket residues around peptide
    receptor_positions = []
    n_residues = len(key_residues)
    
    # Arrange residues in a circle around peptide
    radius = 3
    center_x = 2 + len(sequence) * 0.6
    center_y = peptide_y
    
    for i, (residue, info) in enumerate(key_residues.items()):
        angle = 2 * np.pi * i / n_residues
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Draw receptor residue
        rect = FancyBboxPatch((x - 0.4, y - 0.3), 0.8, 0.6, 
                             boxstyle="round,pad=0.05", 
                             facecolor='lightblue', 
                             edgecolor='darkblue', 
                             linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, residue, ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)
        
        # Draw interaction lines
        interaction_type = info.get('type', 'Unknown')
        
        # Find closest peptide amino acid
        min_dist = float('inf')
        closest_aa_x = 0
        for j, aa in enumerate(sequence):
            aa_x = 2 + j * 1.2
            dist = np.sqrt((x - aa_x)**2 + (y - peptide_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_aa_x = aa_x
        
        # Draw interaction line
        if 'H-bond' in interaction_type:
            line_style = '--'
            line_color = 'blue'
            line_width = 2
        elif 'Electrostatic' in interaction_type:
            line_style = '-'
            line_color = 'red'
            line_width = 2.5
        elif 'Hydrophobic' in interaction_type:
            line_style = ':'
            line_color = 'green'
            line_width = 2
        else:
            line_style = '-'
            line_color = 'gray'
            line_width = 1
        
        ax.plot([closest_aa_x, x], [peptide_y, y], 
               linestyle=line_style, color=line_color, linewidth=line_width, alpha=0.6, zorder=1)
        
        receptor_positions.append((x, y, residue, interaction_type))
    
    # Add title
    ax.text(center_x, 9, f'{pocket_name} Pocket - Peptide Interactions', 
           ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightblue', label=f'{pocket_name} Residues'),
        mpatches.Patch(color='#FF6B6B', label='Acidic AA'),
        mpatches.Patch(color='#4ECDC4', label='Basic AA'),
        mpatches.Patch(color='#FFA07A', label='Polar AA'),
        mpatches.Patch(color='#E0E0E0', label='Special AA'),
    ]
    
    # Add interaction type legend
    from matplotlib.lines import Line2D
    interaction_legend = [
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='H-bond'),
        Line2D([0], [0], color='red', linewidth=2.5, linestyle='-', label='Electrostatic'),
        Line2D([0], [0], color='green', linewidth=2, linestyle=':', label='Hydrophobic'),
    ]
    
    legend1 = ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
                       frameon=True, fancybox=True, shadow=True, title='Amino Acids')
    legend2 = ax.legend(handles=interaction_legend, loc='upper right', fontsize=10,
                       frameon=True, fancybox=True, shadow=True, title='Interactions')
    ax.add_artist(legend1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Interaction diagram saved to {output_file}")
    plt.close()

def generate_3d_structure_pdb(sequence, output_file):
    """Generate 3D structure in PDB format (simplified extended conformation)"""
    print(f"Generating 3D structure for {sequence}...")
    
    # Amino acid properties (approximate coordinates for backbone atoms)
    # This is a simplified extended conformation
    
    with open(output_file, 'w') as f:
        f.write("HEADER    PEPTIDE\n")
        f.write(f"TITLE     {sequence}\n")
        f.write("REMARK    Generated simplified 3D structure\n")
        
        atom_num = 1
        residue_num = 1
        
        for i, aa in enumerate(sequence):
            # Simplified: place each residue 3.8 Å apart along x-axis (extended conformation)
            x = i * 3.8
            y = 0.0
            z = 0.0
            
            # N atom
            f.write(f"ATOM  {atom_num:5d}  N   {aa:3s} A{residue_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           N\n")
            atom_num += 1
            
            # CA atom (alpha carbon)
            f.write(f"ATOM  {atom_num:5d}  CA  {aa:3s} A{residue_num:4d}    {x+1.5:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            atom_num += 1
            
            # C atom (carbonyl carbon)
            f.write(f"ATOM  {atom_num:5d}  C   {aa:3s} A{residue_num:4d}    {x+2.5:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            atom_num += 1
            
            # O atom (carbonyl oxygen)
            f.write(f"ATOM  {atom_num:5d}  O   {aa:3s} A{residue_num:4d}    {x+2.5:8.3f}{y+1.2:8.3f}{z:8.3f}  1.00  0.00           O\n")
            atom_num += 1
            
            residue_num += 1
        
        f.write("END\n")
    
    print(f"3D structure (PDB) saved to {output_file}")

# Main execution
if __name__ == "__main__":
    # Test with EDGET peptide
    sequence = "EDGET"
    
    # Generate 2D structure
    generate_2d_structure(sequence, "/home/ubuntu/peptide_2d_structure.png")
    
    # Generate 3D structure (PDB format)
    generate_3d_structure_pdb(sequence, "/home/ubuntu/peptide_3d_structure.pdb")
    
    # Generate interaction diagrams
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
    
    generate_interaction_diagram(sequence, "T1R1", t1r1_residues, 
                                "/home/ubuntu/peptide_t1r1_interactions.png")
    generate_interaction_diagram(sequence, "T1R3", t1r3_residues,
                                "/home/ubuntu/peptide_t1r3_interactions.png")
    
    print("\nAll structures generated successfully!")
    print("Files created:")
    print("  - peptide_2d_structure.png")
    print("  - peptide_3d_structure.pdb")
    print("  - peptide_t1r1_interactions.png")
    print("  - peptide_t1r3_interactions.png")

