# 3. Example 1: CUDA Streams

Slides 59–65. A stream is a queue of GPU operations that run in the order they are issued. Within a stream, work runs in order. Across streams there is no ordering, so work can overlap.

## `branches.cu`: independent branches, then a join (slides 60–62)

```
x = x1 * x1       # step 1   chain A
y = x2 * x2       # step 2   chain B
z = x * sin(x1)   # step 3   chain A
w = y * cos(x2)   # step 4   chain B
z = z + w         # step 5   waits for both
```

Slide 61: one stream per chain. The 4th launch argument picks the stream, `<<<grid, block, smem, stream>>>`.

```cuda
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
square<<<g,b,0,s1>>>(x1, x, n);
square<<<g,b,0,s2>>>(x2, y, n);
left<<<g,b,0,s1>>>(x1, x, z, n);
right<<<g,b,0,s2>>>(x2, y, w, n);
```

Slide 62: an event is how one stream waits for another.

```cuda
cudaEvent_t ready;
cudaEventCreate(&ready);
cudaEventRecord(ready, s2);          // mark the end of step 4 in s2
cudaStreamWaitEvent(s1, ready, 0);   // s1 waits on the GPU; the host does not block
combine<<<g,b,0,s1>>>(z, w, n);      // step 5 sees both z and w
cudaStreamSynchronize(s1);           // host waits for s1, which now covers both chains
checkCudaErrors(cudaGetLastError());
```

```
$ ./branches
n=1048576 failures=0
PASS streams
```

Try removing `cudaStreamWaitEvent`. Step 5 may then read `w` before step 4 has written it.

## `batches.cu`: copies and computation across batches (slides 63–65)

Each batch runs the slide-63 sequence in its stream:

```cuda
cudaMemcpyAsync(d_in, h_in, bytes,
    cudaMemcpyHostToDevice, s);
kernel<<<grid, block, 0, s>>>(
    d_in, d_out, n);
cudaMemcpyAsync(h_out, d_out, bytes,
    cudaMemcpyDeviceToHost, s);
cudaStreamSynchronize(s);
```

Order within a batch comes from the stream. Overlap comes from using several streams, and from pinned host memory (`cudaMallocHost`), without which async copies do not run asynchronously. The program times 8 batches in one stream, then in 8 streams:

```
$ ./batches                      # RTX 2080 Ti
8 batches in 1 stream(s):   23.75 ms
8 batches in 8 stream(s):   12.81 ms
PASS batches
```

The gain depends on the copy/compute ratio and on the GPU's copy engines. On an H200 MIG 1g.18gb slice it was 28.98 → 26.22 ms.
