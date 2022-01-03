__kernel void RAND_ADD( __global cl_ulong* ARRAY,
                        __global cl_ulong* IDX,
                        cl_ulong iters,
                        cl_ulong pes) {
  cl_ulong i = 0;
  cl_ulong start = 0;

  start = (cl_ulong) (get_global_id(0) * iters);
  for (i=start; i<(start+iters); i++) {
    atom_add(&ARRAY[IDX[i]], (cl_ulong)(0x1));
  }
}

__kernel void RAND_CAS( __global cl_ulong* ARRAY,
                        __global cl_ulong* IDX,
                        cl_ulong iters,
                        cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;

  start = (cl_ulong) (get_global_id(0) * iters);
  for( i=start; i<(start+iters); i++ ){
    atom_cmpxchg(&ARRAY[IDX[i]], ARRAY[IDX[i]], ARRAY[IDX[i]]);
  }
}

__kernel void STRIDE1_ADD( __global cl_ulong* ARRAY,
                           __global cl_ulong* IDX,
                           cl_ulong iters,
                           cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;

  start = (cl_ulong) (get_global_id(0) * iters);
  for( i=start; i<(start+iters); i++ ){
    atom_add(&ARRAY[i], (cl_ulong)(0xF));
  }
}

__kernel void STRIDE1_CAS( __global cl_ulong* ARRAY,
                           __global cl_ulong* IDX,
                           cl_ulong iters,
                           cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;

  start = (cl_ulong) (get_global_id(0) * iters);
  for( i=start; i<(start+iters); i++ ){
    atom_cmpxchg(&ARRAY[i], ARRAY[i], ARRAY[i]);
  }
}

__kernel void STRIDEN_ADD( __global cl_ulong* ARRAY,
                           __global cl_ulong* IDX,
                           cl_ulong iters,
                           cl_ulong pes,
                           cl_ulong stride) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;

  start = (cl_ulong) (get_global_id(0) * iters);
  for( i=start; i<(start+iters); i+=stride ){
    atom_add(&ARRAY[i], (cl_ulong)(0xF));
  }
}

__kernel void STRIDEN_CAS( __global cl_ulong* ARRAY,
                           __global cl_ulong* IDX,
                           cl_ulong iters,
                           cl_ulong pes,
                           cl_ulong stride) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;

  start = (cl_ulong) (get_global_id(0) * iters);
  for( i=start; i<(start+iters); i+=stride ){
    atom_cmpxchg(&ARRAY[i], ARRAY[i], ARRAY[i]);
  }
}

__kernel void PTRCHASE_ADD( __global cl_ulong* ARRAY,
                            __global cl_ulong* IDX,
                            cl_ulong iters,
                            cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;

  start =  (cl_ulong) (get_global_id(0) * iters);
  for( i=0; i<iters; i++ ){
    start = atom_add(&IDX[start], (cl_ulong)(0x00ull));
  }
}

__kernel void PTRCHASE_CAS( __global cl_ulong* ARRAY,
                            __global cl_ulong* IDX,
                            cl_ulong iters,
                            cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;

  start = (cl_ulong)(get_global_id(0)) * iters;
  for( i=0; i<iters; i++ ){
    start = atom_cmpxchg(&IDX[start], start, IDX[start]);
  }
}

__kernel void SG_ADD( __global cl_ulong* ARRAY,
                      __global cl_ulong* IDX,
                      cl_ulong iters,
                      cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;
  cl_ulong src    = 0;
  cl_ulong dest   = 0;
  cl_ulong val    = 0;

  start = (cl_ulong) (get_global_id(0) * iters);
  for( i=start; i<(start+iters); i++ ){
    src = atom_add(&IDX[i], (0x00ull)(0xF));
    dest = atom_add(&IDX[i+1], (cl_ulong)(0x00ull));
    val = atom_add(&ARRAY[src], (cl_ulong)(0x01ull));
    atom_add(&ARRAY[dest], val);
  }
}

__kernel void SG_CAS( __global cl_ulong* ARRAY,
                      __global cl_ulong* IDX,
                      cl_ulong iters,
                      cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;
  cl_ulong src    = 0;
  cl_ulong dest   = 0;
  cl_ulong val    = 0;

  start = (cl_ulong) (get_global_id(0) * iters);
  for( i=start; i<(start+iters); i++ ){
    src = atom_cmpxchg(&IDX[i], src, IDX[i]);
    dest = atom_cmpxchg(&IDX[i+1], dest, IDX[i+1]);
    val = atom_cmpxchg(&ARRAY[src], val, ARRAY[src]);
    atom_cmpxchg(&ARRAY[dest], &ARRAY[dest], val);
  }
}

__kernel void CENTRAL_ADD( __global cl_ulong* ARRAY,
                           __global cl_ulong* IDX,
                           cl_ulong iters,
                           cl_ulong pes) {
  cl_ulong i      = 0;

  for( i=0; i<iters; i++ ){
    atom_add(&ARRAY[0], (cl_ulong)(0x1));
  }
}

__kernel void CENTRAL_CAS( __global cl_ulong* ARRAY,
                           __global cl_ulong* IDX,
                           cl_ulong iters,
                           cl_ulong pes) {
  cl_ulong i      = 0;

  for( i=0; i<iters; i++ ){
    atom_cmpxchg(&ARRAY[0], ARRAY[0], ARRAY[0]);
  }
}

__kernel void SCATTER_ADD( __global cl_ulong* ARRAY,
                           __global cl_ulong* IDX,
                           cl_ulong iters,
                           cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;
  cl_ulong dest   = 0;
  cl_ulong val    = 0;

  start = (cl_ulong) (get_global_id(0) * iters);
  for( i=start; i<(start+iters); i++ ){
    dest = atom_add(&IDX[i+1], (cl_ulong)(0x00ull));
    val = atom_add(&ARRAY[i], (cl_ulong)(0x01ull));
    atom_add(&ARRAY[dest], val);
  }
}

__kernel void SCATTER_CAS( __global cl_ulong* ARRAY,
                           __global cl_ulong* IDX,
                           cl_ulong iters,
                           cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;
  cl_ulong dest   = 0;
  cl_ulong val    = 0;

  start = (cl_ulong) (get_global_id(0) * iters);
  for( i=start; i<(start+iters); i++ ){
    dest = atom_cmpxchg(&IDX[i+1], dest, IDX[i+1]);
    val = atom_cmpxchg(&ARRAY[i], val, ARRAY[i]);
    atom_cmpxchg(&ARRAY[dest], ARRAY[dest], val);
  }
}

__kernel void GATHER_ADD( __global cl_ulong* ARRAY,
                          __global cl_ulong* IDX,
                          cl_ulong iters,
                          cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;
  cl_ulong dest   = 0;
  cl_ulong val    = 0;

  start = (cl_ulong)(get_global_id(0)) * iters;
  for( i=start; i<(start+iters); i++ ){
    dest = atom_add(&IDX[i+1], (cl_ulong)(0x00ull));
    val = atom_add(&ARRAY[dest], (cl_ulong)(0x01ull));
    atom_add(&ARRAY[i], val);
  }
}

__kernel void GATHER_CAS( __global cl_ulong* ARRAY,
                          __global cl_ulong* IDX,
                          cl_ulong iters,
                          cl_ulong pes) {
  cl_ulong i      = 0;
  cl_ulong start  = 0;
  cl_ulong dest   = 0;
  cl_ulong val    = 0;

  start =  (cl_ulong) (get_global_id(0) * iters);
  for( i=start; i<(start+iters); i++ ){
    dest = atom_cmpxchg(&IDX[i+1], dest, IDX[i+1]);
    val = atom_cmpxchg(&ARRAY[dest], val, ARRAY[dest]);
    atom_cmpxchg(&ARRAY[i], ARRAY[i], val);
  }
}
