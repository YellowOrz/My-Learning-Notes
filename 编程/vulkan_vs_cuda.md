
| cuda kernel | GLSL | 
| :------: | :------: |
| threadIdx.x | gl_LocalInvocationID.x | 
| blockIdx.x | gl_WorkGroupId.x |
| threadIdx.x+blockIdx.x*blockDim.x | gl_GlobalInvocationId.x |
| blockDim.x | gl_WorkGroupSize.x |
| __syncthreads() | barrier() |
| __shared__ | shared |
| atomicAdd() | atomicAdd() |
| __shfl_down_sync() | 无 |