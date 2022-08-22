import pdbfixer
import numpy as np
from simtk.openmm import app, LangevinIntegrator, CustomExternalForce, CustomTorsionForce, OpenMMException
from simtk import unit

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms
spring_unit = ENERGY / (LENGTH ** 2)


def refine(input_file, output_file, n=5):
    k1s = [2.5,1,0.5,0.25,0.1]
    k2s = [1,2.5,5,10,25]

    fixer = pdbfixer.PDBFixer(input_file)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    topology, positions = refine_once(fixer.topology, fixer.positions, k1=k1s[0], k2 = k2s[0])

    for i in range(1,n):
        if not (bond_check(topology, positions) or cis_check(topology, positions)):
            print("all failed")
            topology, positions = refine_once(fixer.topology, fixer.positions, k1=k1s[i], k2 = k2s[i])
        elif not bond_check(topology, positions):
            print("bonds failed")
            topology, positions = refine_once(fixer.topology, fixer.positions, k1=k1s[i], k2 = k2s[0])
        elif not cis_check(topology, positions):
            print("CIS failed")
            topology, positions = refine_once(fixer.topology, fixer.positions, k1=k1s[0], k2 = k2s[i])
        else:
            print("all good")
            break
    
    if not stereo_check(topology, positions): 
        print("D-amino acids in model")

    with open(output_file, "w") as out_handle:
        app.PDBFile.writeFile(topology, positions, out_handle, keepIds=True)


def refine_once(topology, positions, k1=1.0, k2=5.0):

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
    
    cis_force = CustomTorsionForce("10*k2*(1+cos(theta))^2")
    cis_force.addGlobalParameter("k2", k2 * ENERGY)

    for chain in modeller.topology.chains():
        residues = [[atom.index for atom in res.atoms() if atom.name in ["N", "CA", "C"]] for res in chain.residues()]
        for i,resi in enumerate(residues[:-1]):
            n_resi = residues[i+1]
            cis_force.addTorsion(resi[1], resi[2], n_resi[0], n_resi[1])
    
    system.addForce(force)
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
        residues = [[atom.index for atom in res.atoms() if atom.name in ["N", "C"]] for res in chain.residues()]
        for i in range(len(residues)-1):
            v = np.linalg.norm(positions[residues[i][1]] -  positions[residues[i+1][0]])
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
        residues = [[atom.index for atom in res.atoms() if atom.name in ["N", "CA", "C"]] for res in chain.residues()]
        for i in range(len(residues)-1):
            p0,p1,p2,p3 = positions[residues[i][1]],positions[residues[i][2]],positions[residues[i+1][0]],positions[residues[i+1][1]]
            if cos_of_torsion(p0,p1,p2,p3) > 0:
                return False
    return True


def stereo_check(topology, positions):
    for residue in topology.residues():
        atom_indices = [atom.index for atom in residue.atoms() if atom.name in ["CA", "N", "C", "CB"]]
        vectors = [positions[i] - positions[atom_indices[0]] for i in atom_indices[1:]]
        if residue.name == "GLY":
            continue

        if np.dot(np.cross(vectors[0], vectors[1]), vectors[2]) > .0*LENGTH**3:
            return False
    return True
