import pdbfixer
import numpy as np
from simtk.openmm import app, LangevinIntegrator, CustomExternalForce, CustomTorsionForce, HarmonicBondForce, OpenMMException
from simtk import unit

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms
spring_unit = ENERGY / (LENGTH ** 2)


def refine(input_file, output_file, n=6):
    k1s = [2.5,1,0.5,0.25,0.1]
    k2s = [2.5,5,7.5,15,25]

    fixer = pdbfixer.PDBFixer(input_file)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    k1 = k1s[0]
    k2 = -1 if cis_check(fixer.topology, fixer.positions) else k2s[0]
    topology, positions = refine_once(fixer.topology, fixer.positions, k1=k1, k2=k2)
    acceptable_bonds, trans_peptide_bonds = bond_check(topology, positions), cis_check(topology, positions)
    correct_chilarity = stereo_check(topology, positions)

    for i in range(1,n-1):

        if not acceptable_bonds:
            print("Bonds failed")
            k1 = k1s[i]
        if not trans_peptide_bonds:
            print("CIS failed")
            k2 = k2s[i]
        if not correct_chilarity:
            print("Stereo failed")
        if acceptable_bonds and trans_peptide_bonds and correct_chilarity:
            break
        else:
            try:
                topology, positions = refine_once(fixer.topology, fixer.positions, k1=k1, k2 = k2)
                acceptable_bonds, trans_peptide_bonds = bond_check(topology, positions), cis_check(topology, positions)
                correct_chilarity = stereo_check(topology, positions)
            except OpenMMException:
                continue

    if not (acceptable_bonds and trans_peptide_bonds and correct_chilarity):
        try:
            print("Final try!!")
            refine_once(topology, positions, k1=.01, k2=-1) # Try one last time with very loose restraints
        except OpenMMException:
            print("Refinemet failed for {}.\nGiving up...".format(output_file))

    with open(output_file, "w") as out_handle:
        app.PDBFile.writeFile(topology, positions, out_handle, keepIds=True)


def refine_once(topology, positions, k1=2.5, k2=2.5):

    # Using amber14 recommended protein force field
    forcefield = app.ForceField("amber14/protein.ff14SB.xml")

    # Fill in the gaps with OpenMM Modeller
    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens(forcefield)

    # Set up force field
    system = forcefield.createSystem(modeller.topology)

    # Keep atoms close to initial prediction
    force = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", k1 * spring_unit)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for residue in modeller.topology.residues():
        for atom in residue.atoms():
            if atom.name in ["CA", "CB", "N", "C"]:
                force.addParticle(atom.index, modeller.positions[atom.index])
    
    system.addForce(force)

    if k2 > 0.0:
        cis_force = CustomTorsionForce("10*k2*(1+cos(theta))^2")
        cis_force.addGlobalParameter("k2", k2 * ENERGY)

        for chain in modeller.topology.chains():
            residues = [res for res in chain.residues()]
            relevant_atoms = [{atom.name:atom.index for atom in res.atoms() if atom.name in ["N", "CA", "C"]} for res in residues]
            for i in range(1,len(residues)):
                if residues[i].name == "PRO":
                    continue

                resi = relevant_atoms[i-1]
                n_resi = relevant_atoms[i]
                cis_force.addTorsion(resi["CA"], resi["C"], n_resi["N"], n_resi["CA"])
        
        system.addForce(cis_force)

    # Set up integrator
    integrator = LangevinIntegrator(0, 0.01, 0.0)

    # Set up the simulation
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # Minimize the energy
    simulation.minimizeEnergy(tolerance = 1*ENERGY)

    return simulation.topology, simulation.context.getState(getPositions=True).getPositions()


def bond_check(topology, positions):
    for chain in topology.chains():
        residues = [{atom.name:atom.index for atom in res.atoms() if atom.name in ["N", "C"]} for res in chain.residues()]
        for i in range(len(residues)-1):
            v = np.linalg.norm(positions[residues[i]["C"]] -  positions[residues[i+1]["N"]])
            if abs(v - 1.329*LENGTH) > 0.1*LENGTH:
                return False
    return True


def cos_of_torsion(p0,p1,p2,p3):
    ab = np.array((p1-p0).value_in_unit(LENGTH))
    cd = np.array((p2-p1).value_in_unit(LENGTH))
    db = np.array((p3-p2).value_in_unit(LENGTH))
    
    u = np.cross(-ab, cd) 
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = np.cross(db, cd)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    
    return (u * v).sum(-1) 
            

def cis_check(topology, positions):
    for chain in topology.chains():
        residues = [res for res in chain.residues()]
        relevant_atoms = [{atom.name:atom.index for atom in res.atoms() if atom.name in ["N", "CA", "C"]} for res in residues]
        for i in range(1,len(residues)):
            if residues[i].name == "PRO":
                continue

            resi = relevant_atoms[i-1]
            n_resi = relevant_atoms[i]
            p0,p1,p2,p3 = positions[resi["CA"]],positions[resi["C"]],positions[n_resi["N"]],positions[n_resi["CA"]]
            if cos_of_torsion(p0,p1,p2,p3) > 0:
                return False
    return True


def stereo_check(topology, positions):
    for residue in topology.residues():
        if residue.name == "GLY":
            continue

        atom_indices = {atom.name:atom.index for atom in residue.atoms() if atom.name in ["N", "CA", "C", "CB"]}
        vectors = [positions[atom_indices[i]] - positions[atom_indices["CA"]] for i in ["N", "C", "CB"]]

        if np.dot(np.cross(vectors[0], vectors[1]), vectors[2]) < .0*LENGTH**3:
            return False
    return True
