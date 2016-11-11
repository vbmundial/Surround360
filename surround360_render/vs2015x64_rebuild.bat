rem call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" amd64

set CURRDIR=%CD%
set BUILD_DIR=%CURRDIR%\build\VS2015\x64
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
cd %BUILD_DIR%

rem set HUNTER_PACKAGE_DIR=D:/projects/hunterPackagesMain
set HUNTER_ROOT=D:/projects/hunterPackages

echo %CD%
set cmake_call=cmake -G "Visual Studio 14 2015 Win64"^
 -DHUNTER_STATUS_DEBUG=ON^
 %CURRDIR% 

echo %cmake_call%
call %cmake_call%

cd ../../..