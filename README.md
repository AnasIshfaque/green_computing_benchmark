# green_computing_benchmark
---
## Training with Libtorch
Run this command if the CMakeLists.txt is modified:
```
cmake .. -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.9/dist-packages/torch
```
Create the executable with this command (run this if the main.cpp is modified):

```
cmake --build . --config Release
```
