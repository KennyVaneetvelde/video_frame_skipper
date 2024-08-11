@echo off
REM Navigate to the project directory
cd /d c:\dev\video_frame_skipper

REM Perform a git pull to update the repository
echo Updating repository...
git pull

REM Activate the virtual environment
call venv\Scripts\activate

REM Prompt user for input and output file paths
set /p INPUTFILE=Enter the path to the input file:
set /p OUTPUTFILE=Enter the path to the output file:

REM Run the Python script with the user-provided paths, ensuring they are properly quoted
python main.py --input_file=%INPUTFILE% --output_file=%OUTPUTFILE%

REM Wait for user input before closing
echo.
echo Script has finished running. Press any key to exit.
pause
