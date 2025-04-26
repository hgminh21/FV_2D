import tecplot as tp
import os
import time

# Connect to Tecplot session
tp.session.connect()

# Set the directory containing the output files
output_dir = './sol/'

# Keep track of the last loaded iteration number
last_loaded_iteration = -1

# Continuously check for new files
while True:
    # Get the list of files in the directory
    files = os.listdir(output_dir)

    # Find all files that match the naming pattern "Q_output_#.dat"
    new_files = [f for f in files if f.startswith("Q_output_") and f.endswith(".dat")]

    # Sort the files numerically based on the iteration number
    new_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

    # Load any new files that have not been loaded yet
    for file_name in new_files:
        # Extract iteration number
        iteration = int(file_name.split('_')[2].split('.')[0])

        if iteration > last_loaded_iteration:
            file_path = os.path.join(output_dir, file_name)
            print(f"Loading time step {iteration}: {file_path}")

            file_path = os.path.abspath(file_path)
            
            # Load the first file with a new dataset
            if iteration == 0:
                tp.macro.execute_command(f"""$!ReadDataSet  '\"{file_path}\"'
                    ReadDataOption = New
                    ResetStyle = No
                    VarLoadMode = ByName
                    AssignStrandIDs = Yes""")
            else:    
                # Load the new dataset without affecting the existing plot/view
                tp.macro.execute_command(f"""$!ReadDataSet  '\"{file_path}\"'
                    ReadDataOption = Append
                    ResetStyle = No
                    VarLoadMode = ByName
                    AssignStrandIDs = Yes""")
                tp.macro.execute_command(f"$!GlobalTime SolutionTime = {iteration}")
                frame = tp.active_frame()
                plot = frame.plot()  # Forces a re-render
                
            # tp.macro.execute_command('$!RedrawAll')
            # Update the last loaded iteration
            last_loaded_iteration = iteration

    # Optionally, wait a short period before checking for new files again
    time.sleep(1)
