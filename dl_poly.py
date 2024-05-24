###############################################################################
# Chris Rumble
# 3/27/22
#
# dl_poly.py is a Python module for allowing easy scripting of DL_POLY
# molecular dynamics simulations. It is largely based on gromacs.py
###############################################################################
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

##################################
# create a trajectory data class #
##################################
@dataclass
class Traj:
    HISTORY:   str   = ''
    comment:   str   = ''
    dt:        float = 0
    n_frame:   int   = 0
    n_atom:    int   = 0
    coord:     float = 0
    atom_name: float = 0
    q:         int   = 0
    
######################
# Parse a trajectory #
######################
def parse_traj(HISTORY='HISTORY'):
    print('Loading trajectory %s' % HISTORY)
    
    # initialize a trajectory class
    traj = Traj()
    traj.HISTORY = HISTORY
    
    # open the HISTORY file
    f_traj = open(HISTORY, 'r')
    
    # get the comment line
    traj.comment = f_traj.readline()
    
    # read the first line of information
    info    = (f_traj.readline()).split()
    traj.n_atom  = int(info[2])
    traj.n_frame = int(info[3])
    
    # read in all frames
    traj.atom_name = list()
    traj.coord     = np.zeros([traj.n_frame, traj.n_atom, 3])
    traj.disp      = np.zeros([traj.n_frame, traj.n_atom], dtype=float)
    traj.mass      = np.zeros([traj.n_atom,  1], dtype=float)
    traj.q         = np.zeros([traj.n_atom,  1], dtype=float)
    traj.box       = np.zeros([traj.n_frame, 3], dtype=float)
    traj.step      = np.zeros([traj.n_frame, 1], dtype=int)
    traj.frame     = np.zeros([traj.n_frame, 1], dtype=int)
    traj.t         = np.zeros([traj.n_frame, 1], dtype=float)

    pbar = tqdm(total=traj.n_frame)
    for i in range(traj.n_frame):
        # read the timestep line
        line = (f_traj.readline()).split()
        traj.step[i]  = line[1]
        traj.frame[i] = i
        traj.t[i]     = line[6]
        
        # read the box size in A
        for j in range(3):
            line     = (f_traj.readline()).split()
            traj.box[i,j] = float(line[j])
        
        # read in atoms in frame
        for j in range(traj.n_atom):
            # read the atom info line
            line = (f_traj.readline()).split()
            if i == 0:
                traj.atom_name.append(line[0])
                traj.mass[j] = line[2]
                traj.q[j]    = line[3]
            traj.disp[i,j] = line[4]
            
            # read the atom's coordinates
            line = (f_traj.readline()).split()
            traj.coord[i,j,:] = np.asarray(line)
        pbar.update(1)
    pbar.close()
    f_traj.close()
    
    return traj

####################
# Run a simulation #
####################
def runsim(par, restart=True, run=True, clust=False, clustpar=None):
    import os
    import subprocess
    import shutil
    
    if clust:
        print('Not yet implemented')
#        if clustpar is None:
#            clustpar = {'partition': 'debug',
#                        'nodes':     1,
#                        'tasks':     1,
#                        'cpus':      16,
#                        'time':      {'day': 0, 'hour': 0, 'min': 15}}

    ############################################
    # Prepare the simulation directory and ini #
    ############################################
    # create a new simulation directory with subdirectory for input files
    print('################################################')
    print('Building DL_POLY simulation: ' + par['name'])
    print('Creating simulation directory...')
    ini = par['name'] + '/ini'
    os.makedirs(ini)
    
    # copy the input files into the ini directory and identify if there's a 
    # checkpoint file (*.chk)
    print('Copying requisite input files...')
    
    # copy the principle input files
    shutil.copy(par['CONTROL'], ini)
    shutil.copy(par['CONFIG'],  ini)
    shutil.copy(par['FIELD'],   ini)

    # determine if the simulation starts from a checkpoint
    if restart:
        try:
            shutil.copy(par['REVIVE'], ini)
        except:
            print('***WARNING***')
            print('Start from checkpoint requested but no \'restart\' given.')
            print('Will try to proceed without REVIVE file.')
            print('***WARNING***')
    
    # copy any extra files
    if 'extras' in par:
        for file in par['extras']:
            shutil.copy(file, ini)
    
    print('Preparations for \'' + par['name'] + '\' are complete!')
    
    ##################
    # Run simulation #
    ##################
    if run:
        # change to the simulation working directory
        os.chdir(par['name'])   
        
        # run the simulation and save stdout and stderr
        print('Running simulation...')
        os.makedirs('./out')
        f   = open('out/run_stdout.txt', 'w')
        e   = open('out/run_stderr.txt', 'w')
        cmd = list()

        if clust:
            print('Not yet implemented')
#            # make the run command with srun parameters
#            cmd.append('srun')
#            cmd.extend(['-N', str(clustpar['nodes'])])
#            cmd.extend(['-n', str(clustpar['tasks'])])
#            cmd.extend(['-c', str(clustpar['cpus'])])
#            cmd.extend(['-p', str(clustpar['partition'])])
#            cmd.extend(['-t', str('%02d' % clustpar['time']['day'])
#                              + '-'
#                              + str('%02d' % clustpar['time']['hour'])
#                              + ':'
#                              + str('%02d' % clustpar['time']['min']) 
#                              + ':00'])
#            cmd.append('gmx_mpi')

        
        # write the mpi commands
        cmd.extend(['mpirun'])
        cmd.extend(['-np', str(par['np'])])
#        cmd.extend(['-bind-to', 'hwthread'])
#        cmd.extend(['--use-hwthread-cpus'])
        cmd.extend([par['exe_path']])
        cmd.extend(['-c', './ini/' + par['CONTROL'].split('/')[-1]])
#        print(cmd)
        
        # run the command we built        
        subprocess.run(cmd, stdout=f, stderr=e, stdin=subprocess.DEVNULL)

        # close our files
        f.close
        e.close
        
        # go back to the original directory
        os.chdir('../')        
        print('DING! Simulation ' + par['name'] + ' is complete!')        

    print('################################################\n\n')
              
    return


##################################################
# eventually use this to make submission scripts #
##################################################
#def gensub(sub, par, check=True, rerun=False, posres=False):
#    import os
#    
#    # open the file
#    f = open(sub['path'], 'w')
#    
#    # write the SLURM parameters
#    f.write('#!/bin/bash\n')
#    f.write('#SBATCH --job-name=' + sub['tpr'] + '\n')
#    f.write('#SBATCH --output=slurm-%J.out\n')
#    f.write('#SBATCH --error=slurm-%J.err\n')
#    f.write('#SBATCH --nodes='  + str(sub['nodes']) + '\n')
#    f.write('#SBATCH --ntasks=' + str(sub['tasks']) + '\n')
#    f.write('#SBATCH --cpus-per-task=' + str(sub['cpus']) + '\n')
#    f.write('#SBATCH --time=' + str('%02d' % sub['time']['day'])
#                              + '-'
#                              + str('%02d' % sub['time']['hour'])
#                              + ':'
#                              + str('%02d' % sub['time']['min']) + ':00\n')
#    f.write('#SBATCH --partition=' + sub['partition'] + '\n')
#    f.write('#SBATCH --extra-node-info=2:8\n')
#    f.write('\n')
#    
#    # load appropriate modules
#    f.write('module load icc/2017.1.132-GCC-6.3.0-2.27  impi/2017.1.132\n')
#    f.write('module load GCC/4.9.3-2.25\n')
#    f.write('module load OpenMPI/1.10.2\n')
#
#
#    f.write('\n')
#    
#    # GROMACS build simulation commands
#    if check:
#        cmd = ['srun', 'gmx_mpi', 'grompp',
#               '-f', './ini/' + os.path.basename(par['runpar']),
#               '-c', './ini/' + os.path.basename(par['box']),
#               '-p', './ini/' + os.path.basename(par['topol']),
#               '-o', par['name']]
#    else:
#        cmd = ['srun', 'gmx_mpi', 'grompp',
#               '-f', './ini/' + os.path.basename(par['runpar']),
#               '-c', './ini/' + os.path.basename(par['box']),
#               '-p', './ini/' + os.path.basename(par['topol']),
#               '-o', par['name']]
#    
#    if posres:
#        cmd.extend(['-r', './ini/' + os.path.basename(par['box'])])
#
#    f.write(' '.join(cmd) + '\n')
#    
#    # write the run command
#    if rerun:
#        f.write('srun gmx_mpi mdrun -ntomp ' + str(sub['cpus']) +
#                ' -v -rerun ' + sub['trr'] + ' -deffnm ' + par['name'] + '\n')
#    else:        
#        f.write('srun gmx_mpi mdrun -ntomp ' + str(sub['cpus']) +
#                ' -v -deffnm ' + sub['tpr'] + '\n')                
#
#    # aaannnnd we're done here
#    f.close()
#    return






























