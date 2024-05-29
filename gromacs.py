###############################################################################
# Chris Rumble
# 4/9/18
#
# gromacs.py is a Python module for allowing easy scripting of GROMACS 
# molecular dynamics simulations.
#
# Current capabilities include:
#   readxvg(filename) for reading *.xvg data files
#   
#   runsim(par) for running a simulation given a dictionary containing paths to
#    the required simulation files
#
#   gensub(sub) for creating SLURM submission scripts to start a simulation
#
# Last update: 9/10/18
###############################################################################
def gensub(sub, par, check=True, rerun=False, posres=False):
    import os
    
    # open the file
    f = open(sub['path'], 'w')
    
    # write the SLURM parameters
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name=' + sub['tpr'] + '\n')
    f.write('#SBATCH --output=slurm-%J.out\n')
    f.write('#SBATCH --error=slurm-%J.err\n')
    f.write('#SBATCH --nodes='  + str(sub['nodes']) + '\n')
    f.write('#SBATCH --ntasks=' + str(sub['tasks']) + '\n')
    f.write('#SBATCH --cpus-per-task=' + str(sub['cpus']) + '\n')
    f.write('#SBATCH --time=' + str('%02d' % sub['time']['day'])
                              + '-'
                              + str('%02d' % sub['time']['hour'])
                              + ':'
                              + str('%02d' % sub['time']['min']) + ':00\n')
    f.write('#SBATCH --partition=' + sub['partition'] + '\n')
    f.write('#SBATCH --extra-node-info=2:8\n')
    f.write('\n')
    
    # load appropriate modules
    f.write('module load icc/2017.1.132-GCC-6.3.0-2.27  impi/2017.1.132\n')
    f.write('module load GCC/4.9.3-2.25\n')
    f.write('module load OpenMPI/1.10.2\n')


    f.write('\n')
    
    # GROMACS build simulation commands
    if check:
        cmd = ['srun', 'gmx_mpi', 'grompp',
               '-f', './ini/' + os.path.basename(par['runpar']),
               '-c', './ini/' + os.path.basename(par['box']),
               '-p', './ini/' + os.path.basename(par['topol']),
               '-o', par['name']]
    else:
        cmd = ['srun', 'gmx_mpi', 'grompp',
               '-f', './ini/' + os.path.basename(par['runpar']),
               '-c', './ini/' + os.path.basename(par['box']),
               '-p', './ini/' + os.path.basename(par['topol']),
               '-o', par['name']]
    
    if posres:
        cmd.extend(['-r', './ini/' + os.path.basename(par['box'])])

    f.write(' '.join(cmd) + '\n')
    
    # write the run command
    if rerun:
        f.write('srun gmx_mpi mdrun -ntomp ' + str(sub['cpus']) +
                ' -v -rerun ' + sub['trr'] + ' -deffnm ' + par['name'] + '\n')
    else:        
        f.write('srun gmx_mpi mdrun -ntomp ' + str(sub['cpus']) +
                ' -v -deffnm ' + sub['tpr'] + '\n')                

    # aaannnnd we're done here
    f.close()
    return


# this is new, try to use it
def read_xvg(filename, prog=False):
    import numpy as np
    from tqdm import tqdm

    # build our dictionary
    data = dict()
    data['comment'] = list()
    data['names']   = list()
    data['x']       = list()
    data['y']       = list()
    
    # parse line by line
    if prog:
        with open(filename, 'r') as f:
            for i, line in tqdm(enumerate(f)):
                if line[0] == '#':
                    data['comment'].append(line)
                elif line.startswith('@'):
                    # ignore lines that don't have a legend label
                    if line.startswith('@ s'):
                        data['names'].append(line.split('"')[1])
                else:
                    data['x'].append(float(line.split()[0]))
                    data['y'].append(line.split()[1:])
    
            # convert the strings of the data into numbers
            for l in range(len(data['x'])):
                data['y'][l] = list(map(float, data['y'][l]))
    else:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if line[0] == '#':
                    data['comment'].append(line)
                elif line.startswith('@'):
                    # ignore lines that don't have a legend label
                    if line.startswith('@ s'):
                        data['names'].append(line.split('"')[1])
                else:
                    data['x'].append(float(line.split()[0]))
                    data['y'].append(line.split()[1:])
    
            # convert the strings of the data into numbers
            for l in range(len(data['x'])):
                data['y'][l] = list(map(float, data['y'][l]))

    # convert the list to a numpy array
    data['x'] = np.asarray(data['x'])
    data['y'] = np.asarray(data['y'])
    
    return data['x'], data['y']

def runsim(par, check=True, run=True, clust=False, clustpar=None, posres=False,
                index=False):
    import os
    import subprocess
    import shutil
    
    if clust:
        if clustpar is None:
            clustpar = {'partition': 'debug',
                        'nodes':     1,
                        'tasks':     1,
                        'cpus':      16,
                        'time':      {'day': 0, 'hour': 0, 'min': 15}}

    ############################################
    # Prepare the simulation directory and ini #
    ############################################
    # create a new simulation directory with subdirectory for input files
    print('################################################')
    print('Building GROMACS simulation: ' + par['name'])
    print('Creating simulation directory...')
    ini = par['name'] + '/ini'
    os.makedirs(ini)
    
    # copy the input files into the ini directory and identify if there's a 
    # checkpoint file (*.chk)
    print('Copying requisite input files...')
    
    # copy the principle input files
    shutil.copy(par['runpar'], ini)
    shutil.copy(par['box'],    ini)
    shutil.copy(par['topol'],  ini)

    # determine if the simulation starts from a checkpoint
    if check:
        try:
            shutil.copy(par['check'], ini)
        except:
            print('***WARNING***')
            print('Start from checkpoint requested but no \'check\' given.')
            print('Will try to proceed without checkpoint file.')
            print('***WARNING***')
    
    # determine if we need to copy an index file
    if index:
        shutil.copy(par['index'], ini)

    # copy any extra files
    if 'extras' in par:
        for file in par['extras']:
            shutil.copy(file, ini)
    
    print('Preparations for \'' + par['name'] + '\' are complete!')
    
    #################################################
    # Create simulation binary and run if requested #
    #################################################
    # NOTE: should think about adding some sort of error catching for 
    # simulations that don't compile or run properly
    if run:
        # change to the simulation working directory
        os.chdir(par['name'])    
    
        # make the grompp command and build the simulation binary
        print('Building the simulation binary...')
        cmd = list()
        
        # set the gmx executable
        if clust:
            cmd.append('srun')
            cmd.append('gmx_mpi')
        else:
            cmd.append(par['gropath'])
        
        # add the grompp argument
        cmd.append('grompp')        
        
        # add the checkpoint argument if requested
        if check:
            cmd.extend(['-t', './ini/' + os.path.basename(par['check'])])
            
        # add the position restraint file is requested (needed in GROMACS2018)
        if posres:
            cmd.extend(['-r', './ini/' + os.path.basename(par['box'])])
       
        # add an index file if requested
        if index:
            cmd.extend(['-n', './ini/' + os.path.basename(par['index'])])

        # add the rest
        cmd.extend(['-f', './ini/' + os.path.basename(par['runpar'])])
        cmd.extend(['-c', './ini/' + os.path.basename(par['box'])])
        cmd.extend(['-p', './ini/' + os.path.basename(par['topol'])])
        cmd.extend(['-o', par['name']])
        
        # build it and save stdout and stderr
        os.mkdir('out')
        f = open('out/build_stdout.txt', 'w')
        e = open('out/build_stderr.txt', 'w')
        subprocess.run(cmd, stdout=f, stderr=e)
        f.close
        e.close
        
        # run the simulation and save stdout and stderr
        print('Running simulation...')
        f = open('out/run_stdout.txt', 'w')
        e = open('out/run_stderr.txt', 'w')
        
        if clust:
            # make the run command with srun parameters
            cmd = list()
            cmd.append('srun')
            cmd.extend(['-N', str(clustpar['nodes'])])
            cmd.extend(['-n', str(clustpar['tasks'])])
            cmd.extend(['-c', str(clustpar['cpus'])])
            cmd.extend(['-p', str(clustpar['partition'])])
            cmd.extend(['-t', str('%02d' % clustpar['time']['day'])
                              + '-'
                              + str('%02d' % clustpar['time']['hour'])
                              + ':'
                              + str('%02d' % clustpar['time']['min']) 
                              + ':00'])
            cmd.append('gmx_mpi')
        else:
            # make the command for local running
            cmd = list()
            cmd.append(par['gropath'])
        
        # add the mdrun stuff
        cmd.extend(['mdrun', '-v', '-deffnm', par['name']])
        
        # specify the number of omp threads if on the cluster
        if clust:
            cmd.extend(['-ntomp', str(clustpar['cpus'])])

            
        # run the command we built        
        subprocess.run(cmd, stdout=f, stderr=e)

        # close our files
        f.close
        e.close
        
        # go back to the original directory
        os.chdir('../')        
        print('DING! Simulation ' + par['name'] + ' is complete!')        

    print('################################################\n\n')
              
    return



###############################################################################

# this is old, try not to use it
def readxvg(filename):
    import numpy as np
    import warnings
    
    warnings.warn('You should not use this version of readxvg, use read_xvg.')
    
    # build our dictionary
    data = dict()
    data['comment'] = list()
    data['names']   = list()
    data['x']       = list()
    data['y']       = list()
    
    # parse line by line
    with open(filename, 'r') as f:
        for i,line in enumerate(f):
            if line[0] == '#':
                data['comment'].append(line)
            elif line.startswith('@'):
                # ignore lines that don't have a legend label
                if line.startswith('@ s'):
                    data['names'].append(line.split('"')[1])
            else:
                data['x'].append(float(line.split()[0]))
                data['y'].append(line.split()[1:])

        # convert the strings of the data into numbers
        for l in range(len(data['x'])):
            data['y'][l] = list(map(float, data['y'][l]))

    # convert the list to a numpy array
    data['x'] = np.asarray(data['x'])
    data['y'] = np.asarray(data['y'])
    return data




























