rem Define here the path to your conda installation
set CONDAPATH=PATH\TO\Anaconda3

rem Define here the name of the environment
set ENVNAME=watanabe-env

rem Remove environment if it already exists
call conda env remove --name %ENVNAME% -y

rem Create the environment
call conda create --name %ENVNAME% python=3.9 -y --copy

rem Activate the conda environment
call %CONDAPATH%\Scripts\activate.bat %ENVNAME%

call pip install rdkit mordred Boruta padelpy

call pip install numpy==1.19.5 pandas matplotlib joblib black

rem Deactivate the environment
call conda deactivate