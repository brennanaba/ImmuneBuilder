import pdbfixer
import numpy as np
from simtk.openmm import app, LangevinIntegrator, CustomExternalForce, CustomTorsionForce, HarmonicBondForce, OpenMMException
from simtk import unit

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms
spring_unit = ENERGY / (LENGTH ** 2)


def refine(input_file, output_file, n=6):
    k1s = [2.5,1,0.5,0.25,0.1,0.001]
    k2s = [2.5,5,7.5,15,25,50]

    fixer = pdbfixer.PDBFixer(input_file)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    k1 = k1s[0]
    k2 = -1 if cis_check(fixer.topology, fixer.positions) else k2s[0]

    for i in range(n):
        try:
            simulation = refine_once(fixer.topology, fixer.positions, k1=k1, k2 = k2)
            topology, positions = simulation.topology, simulation.context.getState(getPositions=True).getPositions()
            acceptable_bonds, trans_peptide_bonds = bond_check(topology, positions), cis_check(topology, positions)
        except OpenMMException as e:
            if (i == n-1) and ("positions" not in locals()):
                print("OpenMM failed to refine {}".format(output_file))
                raise e
            else:
                continue

        # If peptide bonds are the wrong length, decrease the strength of the positional restraint
        if not acceptable_bonds:
            k1 = k1s[min(i, len(k1s)-1)]

        # If there are still cis isomers in the model, increase the force to fix these
        if not trans_peptide_bonds:
            k2 = k2s[min(i, len(k2s)-1)]
        
        # If peptide bond lengths and torsions are okay, check and fix the chirality.
        if acceptable_bonds and trans_peptide_bonds:
            simulation = chirality_fixer(simulation)
            topology, positions = simulation.topology, simulation.context.getState(getPositions=True).getPositions()

            # If it passes all the tests, we are done
            if bond_check(topology, positions) and cis_check(topology, positions) and stereo_check(topology, positions):
                break

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
    simulation.minimizeEnergy()

    return simulation


def chirality_fixer(simulation):
    topology = simulation.topology
    positions = simulation.context.getState(getPositions=True).getPositions()
    
    d_stereoisomers = []
    for residue in topology.residues():
        if residue.name == "GLY":
            continue

        atom_indices = {atom.name:atom.index for atom in residue.atoms() if atom.name in ["N", "CA", "C", "CB"]}
        vectors = [positions[atom_indices[i]] - positions[atom_indices["CA"]] for i in ["N", "C", "CB"]]

        if np.dot(np.cross(vectors[0], vectors[1]), vectors[2]) < .0*LENGTH**3:
            # If it is a D-stereoisomer then flip its H atom
            indices = [x.index for x in residue.atoms() if x.name in ["HA", "CA"]]
            positions[indices[0]] = 2*positions[indices[1]] - positions[indices[0]]
            
            # Fix the H atom in place
            particle_mass = simulation.system.getParticleMass(indices[0])
            simulation.system.setParticleMass(indices[0], 0.0)
            d_stereoisomers.append((indices[0], particle_mass))
            
    if len(d_stereoisomers) > 0:
        simulation.context.setPositions(positions)

        # Minimize the energy with the evil hydrogens fixed
        simulation.minimizeEnergy()

        # Minimize the energy letting the hydrogens move
        for atom in d_stereoisomers:
            simulation.system.setParticleMass(*atom)
        simulation.minimizeEnergy()
    
    return simulation


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
